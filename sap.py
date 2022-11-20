import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
import dsp
import util

def sap2ba(alphas, ds):
    Bs = torch.stack([alphas, torch.ones_like(alphas)], -1)
    As = torch.flip(Bs, (-1,))
    ds = torch.stack([torch.zeros_like(ds), ds], -1)
    return Bs, As, ds

def flip(Bs, ds = None):
    if ds is not None:
        ds = torch.max(ds, -1, keepdim=True)-ds
        ds, Bs = torch.flip(ds, (-1,)), torch.flip(Bs, (-1,))
        return Bs, ds
    else:
        Bs = torch.flip(Bs, (-1,))
        return Bs

class SAP(nn.Module):
    def __init__(self,
                 num_sample = 16000):
        super(SAP, self).__init__()
        self.num_sample = num_sample

    def forward(self, z):
        alphas, ds = z['alphas'], z['ds']
        Bs, As, ds = sap2ba(alphas, ds)
        sap = dsp.IIR(Bs, As, ds = ds, n = self.num_sample)
        return sap

class SAP_Decoder(nn.Module):
    def __init__(self,
                 input_shape,
                 num_channel = 8,
                 sap_per_channel = 4,
                 max_ds_len = 500,
                 st_prime = False,
                 ):

        super().__init__()

        output_shape = (num_channel, sap_per_channel)
        self.alphas_decoder = util.LinearLinear(input_shape, output_shape)
        self.ds_const = nn.Parameter(torch.tensor([[[131, 151, 337, 353],
                                                    [103, 173, 331, 373],
                                                    [89, 181, 307, 401],
                                                    [79, 197, 281, 419],
                                                    [61, 211, 257, 431],
                                                    [47, 229, 251, 443]]]),
                                     requires_grad = False)


    def forward(self, latent):
        alphas = self.alphas_decoder(latent)
        alphas = torch.sigmoid(alphas)
        params = dict(alphas = alphas,
                      ds = self.ds_const)
        return params
