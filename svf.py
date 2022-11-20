import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
import numpy as np

import dsp
import util

def svf2ba(twoRs, Gs, c_hps, c_bps, c_lps):
    b0 = c_hps+c_bps*Gs+c_lps*(Gs**2)
    b1 = -c_hps*2+c_lps*2*(Gs**2)
    b2 = c_hps-c_bps*Gs+c_lps*(Gs**2)

    a0 = 1+(Gs**2)+twoRs*Gs
    a1 = 2*(Gs**2)-2
    a2 = 1+(Gs**2)-twoRs*Gs

    Bs = torch.stack([b0, b1, b2], -1)
    As = torch.stack([a0, a1, a2], -1)
    return Bs, As

class SVF(nn.Module):
    def __init__(self, num_sample=4000):
        super().__init__()
        self.num_sample = num_sample
        self.arange = torch.arange(3) 
        self.delays = nn.Parameter(dsp.delay(self.arange, n=num_sample), requires_grad=False)

    def forward(self, z):
        twoRs = z['twoRs']
        Gs = z['Gs']
        c_hps, c_bps, c_lps = z['c_hps'], z['c_bps'], z['c_lps']
        Bs, As = svf2ba(twoRs, Gs, c_hps, c_bps, c_lps)
        svf = dsp.IIR(Bs, As, delays=self.delays)
        return svf

class SVF_Decoder(nn.Module):
    def __init__(self,
                 input_shape=(500, 64), # excludes batch
                 num_channel=50,
                 svf_per_channel=16,
                 sr=48000,
                 f_init=(40, 12000),
                 parallel=True,
                 advanced=False):
        super().__init__()
        self.parallel = parallel
        self.svf_per_channel = svf_per_channel
        self.advanced = advanced
        self.sr = sr

        output_shape = (num_channel, svf_per_channel)

        self.Gs_decoder = util.LinearLinear(input_shape, output_shape) 
        self.twoRs_decoder = util.LinearLinear(input_shape, output_shape)
        self.c_hps_decoder = util.LinearLinear(input_shape, output_shape)
        self.c_bps_decoder = util.LinearLinear(input_shape, output_shape)
        self.c_lps_decoder = util.LinearLinear(input_shape, output_shape)

        Gs_bias = f_init[0]*((f_init[1]/f_init[0])**torch.linspace(0, 1, svf_per_channel)) 
        if self.advanced: Gs_bias = Gs_bias[torch.randperm(Gs_bias.shape[-1])]
        Gs_bias = nn.Parameter(torch.log(2*Gs_bias/(sr-2*Gs_bias)))
        self.Gs_decoder.dense_2.bias = Gs_bias

        if not parallel:
            c_bias = torch.ones(svf_per_channel)
            self.c_hps_decoder.dense_2.bias = nn.Parameter(c_bias)
            self.c_bps_decoder.dense_2.bias = nn.Parameter(c_bias*2)
            self.c_lps_decoder.dense_2.bias = nn.Parameter(c_bias)

    def forward(self, latent):
        Gs = self.Gs_decoder(latent)
        if not self.parallel:
            latent = latent / np.sqrt(self.svf_per_channel)
        twoRs = self.twoRs_decoder(latent)
        c_hps = self.c_hps_decoder(latent)
        c_bps = self.c_bps_decoder(latent)
        c_lps = self.c_lps_decoder(latent)

        Gs = torch.tan(0.5*torch.sigmoid(Gs)*np.pi)
        twoRs = 2*F.softplus(twoRs)/F.softplus(torch.zeros(1)).item()+1e-2

        params = dict(Gs=Gs, twoRs=twoRs, c_hps=c_hps, c_bps=c_bps, c_lps=c_lps)
        return params

class PEQ_Decoder(nn.Module):
    def __init__(self,
                 input_shape=(500, 64), 
                 num_channel=50,
                 svf_per_channel=16,
                 sr=48000,
                 f_init=(40, 12000),
                 parallel=True):
        super().__init__()

        self.parallel = parallel
        self.svf_per_channel = svf_per_channel
        self.sr = sr

        output_shape = (num_channel, svf_per_channel)

        self.Gs_decoder = util.LinearLinear(input_shape, output_shape) 
        self.twoRs_decoder = util.LinearLinear(input_shape, output_shape)
        self.cs_decoder = util.LinearLinear(input_shape, output_shape)

        Gs_bias = f_init[0]*((f_init[1]/f_init[0])**torch.linspace(0, 1, svf_per_channel)) 
        Gs_bias = nn.Parameter(torch.log(2*Gs_bias/(sr-2*Gs_bias)))
        self.Gs_decoder.dense_2.bias = Gs_bias

        self.cs_decoder.dense_2.bias = nn.Parameter(torch.ones(svf_per_channel)*2)

    def forward(self, latent):
        Gs = self.Gs_decoder(latent)
        if not self.parallel: latent = latent/np.sqrt(self.svf_per_channel)

        twoRs = self.twoRs_decoder(latent)
        cs = self.cs_decoder(latent)
        cs = 10**(-F.softplus(cs-3))

        Gs = torch.tan(0.5*torch.sigmoid(Gs)*np.pi) 
        twoRs = 2*F.softplus(twoRs)/F.softplus(torch.zeros(1)).item()
        twoRs = torch.cat([twoRs[..., :1]+np.sqrt(2), 
                           twoRs[..., 1:-1], 
                           twoRs[..., -1:]+np.sqrt(2)], -1)

        c_hps_bell = torch.ones_like(cs[..., 1:-1])
        c_bps_bell = twoRs[..., 1:-1]*cs[..., 1:-1]
        c_lps_bell = torch.ones_like(cs[..., 1:-1])

        c_hps_ls = torch.ones_like(cs[..., :1])
        c_bps_ls = twoRs[..., :1]*torch.sqrt(cs[..., :1])
        c_lps_ls = cs[..., :1]

        c_hps_hs = cs[..., -1:]
        c_bps_hs = twoRs[..., -1:]*torch.sqrt(cs[..., -1:])
        c_lps_hs = torch.ones_like(cs[..., -1:])

        c_hps = torch.cat([c_hps_ls, c_hps_bell, c_hps_hs], -1)
        c_bps = torch.cat([c_bps_ls, c_bps_bell, c_bps_hs], -1)
        c_lps = torch.cat([c_lps_ls, c_lps_bell, c_lps_hs], -1)

        params = dict(Gs=Gs, twoRs=twoRs, c_hps=c_hps, c_bps=c_bps, c_lps=c_lps, cs=cs)
        return params
