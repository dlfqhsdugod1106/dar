import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio 
import numpy as np
from itertools import islice
from tqdm import tqdm
from scipy import signal

import dsp
import util
import scipy.linalg

class FDN(nn.Module):
    def __init__(self,
                 num_channel = 8,
                 num_sample = 16000,):
        super(FDN, self).__init__()

        self.num_channel = num_channel
        self.num_sample = num_sample
        self.I = nn.Parameter(torch.eye(self.num_channel), requires_grad = False)

    def forward(self, z):
        ds = z['ds']
        fb_adm, fb_gain, fb_filter = z['fb_adm'], z['fb_gain'], z['fb_filter']
        pre_gain, post_gain = z['pre_gain'], z['post_gain']
        pre, post = z['pre'], z['post']
        fir = z['fir']

        batch_size = pre_gain.shape[0]

        fb_mix = self.scatter(fb_adm).unsqueeze(-3)

        if fb_filter != None:
            fb_line = fb_gain.unsqueeze(-2) * fb_filter
        else:
            fb_line = fb_gain.unsqueeze(-2)
        fb_mat = fb_mix * fb_line.unsqueeze(-2)

        delays = dsp.delay(ds, self.num_sample).transpose(-1, -2)
        feedback = delays.unsqueeze(-1) * util.inverse(self.I - fb_mat * delays.unsqueeze(-2))

        if pre != None:
            if pre.shape[-2] == 1:
                fb_and_pre = torch.sum(feedback * pre_gain.view(batch_size, 1, 1, self.num_channel), -1, keepdim = True) * pre
            else:
                fb_and_pre = util.matmul(feedback, pre)
        else:
            fb_and_pre = torch.sum(feedback * pre_gain.view(batch_size, 1, 1, self.num_channel), -1, keepdim = True)

        if post != None:
            if post.shape[-1] == 1:
                fdn = torch.sum(fb_and_pre * post_gain.view(batch_size, 1, self.num_channel, 1), -2, keepdim = True) * post
            else:
                fdn = util.matmul(post, fb_and_pre)
        else:
            fdn = torch.sum(fb_and_pre * post_gain.view(batch_size, 1, self.num_channel, 1), -2, keepdim = True)

        fdn = fdn.squeeze(-1).squeeze(-1)
        fdn = torch.fft.irfft(fdn, n = self.num_sample)

        fir = F.pad(fir, (0, fdn.shape[-1] - fir.shape[-1]))
        fdn = fdn + fir
        return fdn

    def scatter(self, adm):
        const = 2 / torch.sum(adm ** 2, -1, keepdim = True).unsqueeze(-1)
        scat = const * torch.matmul(adm.unsqueeze(-1), adm.unsqueeze(-2))
        scat = scat - self.I
        return scat


class FDN_Decoder(nn.Module):
    def __init__(self,
                 input_shape,
                 num_channel = 8,
                 max_ds_len = 2000,
                 fir_len = 100,
                 trainable_admittance = False
                 ):

        super().__init__()

        self.max_ds_len = max_ds_len

        self.trainable_admittance = trainable_admittance
        self.fb_adm_decoder = util.LinearLinear(input_shape, (num_channel, 1))
        self.fb_gain_decoder = util.LinearLinear(input_shape, (num_channel, 1))
        self.pre_gain_decoder = util.LinearLinear(input_shape, (num_channel, 1))
        self.post_gain_decoder = util.LinearLinear(input_shape, (num_channel, 1))
        self.fir_decoder = util.LinearLinear(input_shape, (fir_len, 1))

        self.ds_const = nn.Parameter(torch.tensor([[233, 311, 421, 461, 587, 613, 131, 98]])[:, :num_channel], 
                                        requires_grad = False)

    def forward(self, latent):
        fb_gain = self.fb_gain_decoder(latent).squeeze(-1)
        fb_gain = F.softplus(fb_gain)
        fb_gain = fb_gain / 1e1
        fb_gain = 10 ** (-fb_gain)
        fb_gain = torch.ones_like(fb_gain) * (1 - 1e-5)

        if self.trainable_admittance:
            fb_adm = self.fb_adm_decoder(latent).squeeze(-1)
            fb_adm = F.softplus(fb_adm)
        else:
            fb_adm = torch.ones_like(fb_gain)

        pre_gain = self.pre_gain_decoder(latent).squeeze(-1)
        post_gain = self.post_gain_decoder(latent).squeeze(-1)

        fir = self.fir_decoder(latent).squeeze(-1)

        params = dict(ds = self.ds_const,
                      fb_adm = fb_adm, fb_gain = fb_gain,
                      pre_gain = pre_gain, post_gain = post_gain,
                      fir = fir)
        return params
