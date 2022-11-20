import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from nnAudio import features

class SpectralLoss(nn.Module):
    def __init__(self, n_fft, sr=48000, overlap=0.75, eps=1e-7):
        super().__init__()
        self.spec = features.STFT(n_fft=n_fft, sr=sr, hop_length=int(n_fft*(1-overlap)), freq_scale='log',
                                  fmin=20, fmax=sr/2, output_format='Magnitude')
               
    def forward(self, pred, true):
        PRED, TRUE = self.spec(pred), self.spec(true)
        loss = F.l1_loss(PRED, TRUE)
        return loss

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, n_ffts=[256, 512, 1024, 2048, 4096], **kwargs):
        super().__init__()
        self.losses = nn.ModuleList([SpectralLoss(n_fft, **kwargs) for n_fft in n_ffts])

    def forward(self, x_pred, x_true):
        x_pred = x_pred[..., : x_true.shape[-1]]
        losses = [loss(x_pred, x_true) for loss in self.losses]
        return sum(losses).sum()

class DecayRegularizationLoss(nn.Module):
    def __init__(self, tail=500):
        super().__init__()
        self.tail = tail

    def forward(self, ir, eps=1e-12):
        batch_size = ir.shape[0]
        front, rear = ir[..., :self.tail].abs().sum(-1), ir[..., -self.tail:].abs().sum(-1)
        ratio = rear/(front+eps)
        weighted = ratio*F.softmax(ratio, -1)
        loss = weighted.sum(-1).mean()
        return loss

class ParameterMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true, flag = ''):
        losses = {}
        for key in params_true.keys():
            pred, true = params_pred[key], params_true[key]
            if key == 'fb_gain': pred, true = torch.log10(1-pred+1e-7), torch.log10(1-true+1e-7)
            if key == 'g': pred, true = torch.log10(pred+1e-7), torch.log10(true+1e-7)
            if key in ['Gs', 'twoRs', 'fir']: 
                alpha = 10
            elif key == 'cs':
                alpha = 100
            else: 
                alpha = 1
            losses[key] = alpha*F.l1_loss(pred, true)
        loss = sum(list(losses.values()))
        return loss

class LossHandler(nn.Module):
    def __init__(self, losses=['spectral']):
        super().__init__()
        if 'spectral' in losses:
            self.spectral_loss = MultiScaleSpectralLoss()
        if 'decay' in losses:
            self.decay_loss = DecayRegularizationLoss()
        if 'parameter' in losses:
            self.parameter_loss = ParameterMatchLoss()

    def forward(self, pred_ir=None, true_ir=None, pred_param=None, true_param=None):

