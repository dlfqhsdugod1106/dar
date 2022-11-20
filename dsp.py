import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio 
import numpy as np
import util

def convolve(x, y):
    x_pad = F.pad(x, (0, y.shape[-1]-1))
    y_pad = F.pad(y, (0, x.shape[-1]-1))
    X = torch.fft.rfft(x_pad)
    Y = torch.fft.rfft(y_pad)
    conv = torch.fft.irfft(X * Y, x.shape[-1] + y.shape[-1] - 1)
    return conv

def convolves(x):
    x_pad = F.pad(x, (0, (x.shape[-1] - 1) * x.shape[-2]))
    X = torch.fft.rfft(x_pad)
    conv = torch.fft.irfft(util.prod(X), x.shape[-1] + (x.shape[-1] - 1) * x.shape[-2])
    return conv

def delay(ds, n):
    idxs = torch.arange(n // 2 + 1, device = ds.device).unsqueeze(-2)
    phase = ds.unsqueeze(-1) * idxs / n * 2 * np.pi
    delay = torch.view_as_complex(torch.stack([torch.cos(phase), -torch.sin(phase)], -1))
    return delay

def IIR(Bs, As, delays = None, n = None, ds = None, eps = 1e-10):
    if delays is None:
        if ds is None:
            ds = torch.arange(Bs.shape[-1], device = Bs.device)
        delays = delay(ds, n)
    Bs, As = Bs.unsqueeze(-1), As.unsqueeze(-1)
    IIR = torch.sum(Bs * delays, -2) / (torch.sum(As * delays, -2) + eps)
    return IIR

def feedback(H_in, H_out):
    feedback = H_in/(1-H_out*H_in)
    return feedback

def mat_feedback(H_in, H_out, I=None):
    if I is None: I = torch.eye(H_in.shape[-1], device=H_in.device)
    feedback = torch.matmul(H_in, torch.inverse(I-torch.matmul(H_out, H_in)))
    return feedback

def overlap_add(frames, stride_length):
    batch_shape = frames.shape[:-2]
    frame_sig_shape = frames.shape[-2:]
    frames = frames.view((-1,)+frame_sig_shape)
    
    frame_len, device = frames.shape[-1], frames.device
    overlap_add_filter = torch.eye(frame_len, device=device).unsqueeze(-2)
    overlap_add_signal = nn.functional.conv_transpose1d(frames.transpose(-1, -2), 
                                                        overlap_add_filter, 
                                                        stride=stride_length, 
                                                        padding=0).squeeze(-2)

    overlap_add_signal = overlap_add_signal.view(batch_shape+(-1,))
    return overlap_add_signal
