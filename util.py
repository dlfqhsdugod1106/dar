import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
import numpy as np

def cumprod(x):
    mat = torch.tril(torch.ones(x.shape[-1], x.shape[-1], 
                                dtype = torch.cfloat, 
                                device = x.device))
    mat2 = torch.triu(torch.ones(x.shape[-1], x.shape[-1], 
                                dtype = torch.cfloat, 
                                device = x.device), 
                      diagonal = 1)
    mat = mat.view((1,) * (x.ndim - 1) + (x.shape[-1], x.shape[-1]))
    mat2 = mat2.view((1,) * (x.ndim - 1) + (x.shape[-1], x.shape[-1]))
    pre_cumprod = mat * x.unsqueeze(-2) + mat2
    cumprod = prod(pre_cumprod)
    return cumprod

def repeat_interleave(x, repeat):
    shape = x.shape
    ndim = x.ndim
    x = x.unsqueeze(-1).repeat((1,) * ndim + (repeat,))
    x = x.view(shape[:-1] + (-1,))
    return x

def prod(x):
    prod = x[..., 0]
    for i in range(1, x.shape[-1]):
        prod = prod * x[..., i]
    return prod

def inverse(x):
    x = torch.view_as_real(x)
    xl = torch.cat(
        [torch.cat([x[..., 0], -1. * x[..., 1]], -1),
         torch.cat([x[..., 1], x[..., 0]], -1)], 
        -2
    )
    
    xl_inv = torch.inverse(xl)
    
    size = x.shape[-2]
    
    re_inv = xl_inv[..., :size, :size]
    im_inv = xl_inv[..., size:, :size]
    
    inverse = torch.stack((re_inv, im_inv), -1)        
    inverse = torch.view_as_complex(inverse)
    return inverse

def matmul(x, y):
    x, y = torch.view_as_real(x), torch.view_as_real(y)
    re = torch.matmul(x[..., 0], y[..., 0]) - torch.matmul(x[..., 1], y[..., 1])
    im = torch.matmul(x[..., 0], y[..., 1]) + torch.matmul(x[..., 1], y[..., 0])
        
    matmul = torch.stack((re, im), -1)    
    matmul = torch.view_as_complex(matmul)
    return matmul

def cal_rms(amp, eps = 1e-7):
    return np.sqrt(np.mean(np.square(amp), axis=-1) + eps)

def rms_normalize(wav, ref_dBFS=-23.0, eps = 1e-7):
    rms = cal_rms(wav)
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    gain = ref_linear / (rms + eps)
    wav = gain * wav
    return wav    

class LinearLinear(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.dense_1 = nn.Linear(input_shape[0], output_shape[0])
        self.dense_2 = nn.Linear(input_shape[1], output_shape[1])

    def forward(self, x):
        x = self.dense_1(x.transpose(-1, -2))
        x = self.dense_2(x.transpose(-1, -2))
        return x
