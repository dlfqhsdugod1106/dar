import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import features
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, sr=48000, n_fft=1024):
        super().__init__()
        self.spec = features.STFT(n_fft=n_fft, 
                                  hop_length=n_fft//8, 
                                  freq_scale='log',
                                  sr=sr, 
                                  fmin=10,
                                  fmax=sr/2, 
                                  output_format='Magnitude')
        self.conv = nn.Sequential(nn.Conv2d(1, 64, (7, 5), (1, 2)), nn.ReLU(),
                                  nn.Conv2d(64, 128, 5, (2, 1)), nn.ReLU(),
                                  nn.Conv2d(128, 128, 5, 2), nn.ReLU(),
                                  nn.Conv2d(128, 128, 5, 2), nn.ReLU(),
                                  nn.Conv2d(128, 128, 5, 1), nn.ReLU())
        self.fgru = nn.GRU(128, 64, 2, batch_first=True, bidirectional=True)
        self.tgru = nn.GRU(7168, 128, 1, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(torch.log(self.spec(x)+1e-7), 'b f t -> b 1 f t')
        x = rearrange(self.conv(x), 'b c f t -> (b t) f c')
        x = rearrange(self.fgru(x)[0], '(b t) f c -> b t (f c)', b=b)
        x = self.tgru(x)[0]
        x = self.mlp(x)
        print(x.shape)
        return x
