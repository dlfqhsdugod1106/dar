import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
from torch.utils.data import Dataset 
import numpy as np

from train_reverb.dataset.rir_dataset.augment_3 import rdeq
import components.core_dsp_complex as dsp

class rdnoise(Dataset):
    def __init__(self, noise_len = 16000, batch_size = 8, num_batch = 1000, band = 'linear', order = 4):
        super().__init__()

        self.noise_len = noise_len
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.band = band
        self.order = order

    def __getitem__(self, idx):
        noise = torch.rand(self.noise_len)
        rdnoise = rdeq(noise, fb_band = self.band, geq_gain_range = (-12, 12), order = self.order)
        return noise, rdnoise

    def __len__(self):
        return self.num_batch * self.batch_size


class linmagrdnoise(Dataset):
    def __init__(self, noise_len = 16000, batch_size = 8, num_batch = 1000, nbank = 10, repeat = 1):
        super().__init__()

        self.noise_len = noise_len
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.nbank = nbank
        self.repeat = repeat

    def __getitem__(self, idx):

        noise = torch.rand(self.noise_len)
        banks = torch.rand(self.nbank)
        banks = torch.repeat_interleave(banks, self.repeat)
        banks = torch.stack([banks, torch.zeros_like(banks)], -1)
        ir = torch.irfft(banks.unsqueeze(0), 1).squeeze(0)
        ir = ir.numpy()
        noise = noise.numpy()
        rdnoise = np.convolve(ir, noise)[:-ir.shape[-1] + 1]
        return noise, rdnoise, ir

    def __len__(self):
        return self.num_batch * self.batch_size


class melmagrdnoise(Dataset):
    def __init__(self,
                 noise_len = 16000,
                 num_batch = 1000,
                 sr = 16000,
                 num_channel = 1,
                 nbank = 4096,
                 n_mels = 64,
                 db_range = (-12, 12),
                 smoothing = 1,
                 ):
        super().__init__()
        
        self.noise_len = noise_len
        self.num_batch = num_batch
        self.nbank = nbank
        self.sr = sr
        self.num_channel = num_channel
        self.n_mels = n_mels
        self.min_db, self.max_db = db_range[0], db_range[1]
        
        self.melmat = torchaudio.functional.create_fb_matrix(n_freqs = nbank,
                                                             f_min = 0, 
                                                             f_max = self.sr / 2, 
                                                             n_mels = self.n_mels, 
                                                             sample_rate = self.sr)

        self.smoothing = smoothing
        self.smoother = torch.hann_window(2 * self.smoothing + 1)
        self.smoother = self.smoother / torch.sum(self.smoother).view(1, -1)
        
    def __getitem__(self, idx):
        #noise = torch.rand(self.num_channel, self.noise_len)
        noise = torch.rand(self.num_channel, self.noise_len) * 2 - 1
        db_rand = torch.rand(self.num_channel, self.n_mels) * (self.max_db - self.min_db) + self.min_db
        #db_rand = dsp.convolve(db_rand, self.smoother)[:, self.smoothing:-self.smoothing]
        gain_rand = 10 ** (db_rand / 20)
        mel_gain = torch.sum(self.melmat * gain_rand.unsqueeze(-2), -1)
        phase = torch.rand_like(mel_gain)
        banks = torch.stack([mel_gain * torch.cos(phase), mel_gain * torch.sin(phase)], -1)
        ir = torch.irfft(banks.unsqueeze(0), 1).squeeze(0)
        rdnoise = dsp.convolve(ir, noise)[:, :-ir.shape[-1] + 1]
        return noise.squeeze(), rdnoise.squeeze(), ir.squeeze()

    def __len__(self):
        return self.num_channel * self.num_batch



class linspecnoise(Dataset):
    def __init__(self,
                 noise_len = 16000,
                 num_batch = 1000,
                 sr = 16000,
                 num_channel = 8,
                 n_fft = 1024,
                 n_bank = 64,
                 db_range = (-24, 24),
                 smoothing = 1,
                 ):
        super().__init__()
        
        self.noise_len = noise_len
        self.num_batch = num_batch
        self.n_fft = n_fft
        self.sr = sr
        self.num_channel = num_channel
        self.n_bank = n_bank
        self.bank_len = self.n_fft // self.n_bank
        self.min_db, self.max_db = db_range[0], db_range[1]
        
        self.spec = torchaudio.transforms.Spectrogram(n_fft = self.n_fft)
        self.invs = torchaudio.transforms.GriffinLim(n_fft = self.n_fft)
        
        self.linmat = F.conv_transpose1d(torch.eye(self.n_bank).view(self.n_bank, 1, self.n_bank),
                                         torch.bartlett_window(self.bank_len).view(1, 1, -1), 
                                         stride = self.bank_len // 2).squeeze(-2)

        self.smoothing = smoothing
        self.smoother = torch.hann_window(2 * self.smoothing + 1)
        self.smoother = self.smoother / torch.sum(self.smoother).view(1, -1)
        
    def __getitem__(self, idx):
        noise = torch.rand(self.num_channel, self.noise_len + self.n_fft)
        db_rand = torch.rand(self.num_channel, self.n_bank) * (self.max_db - self.min_db) + self.min_db
        db_rand = dsp.convolve(db_rand, self.smoother)[:, self.smoothing:-self.smoothing]
        gain_rand = 10 ** (db_rand / 20)
        lin_gain = torch.sum(self.linmat * gain_rand.unsqueeze(-1), -2)[:, :self.n_fft // 2 + 1]
        rdnoise = self.invs(self.spec(noise) * lin_gain.unsqueeze(-1))
        return noise[:, :self.noise_len].squeeze(), rdnoise[:, :self.noise_len].squeeze()

    def __len__(self):
        return self.num_channel * self.num_batch
    

class melspecnoise(Dataset):
    def __init__(self,
                 noise_len = 16000,
                 num_batch = 1000,
                 sr = 16000,
                 num_channel = 8,
                 n_fft = 4096,
                 n_mels = 64,
                 db_range = (-12, 12),
                 smoothing = 1,
                 ):
        super().__init__()
        
        self.noise_len = noise_len
        self.num_batch = num_batch
        self.n_fft = n_fft
        self.sr = sr
        self.num_channel = num_channel
        self.n_mels = n_mels
        self.min_db, self.max_db = db_range[0], db_range[1]
        
        self.spec = torchaudio.transforms.Spectrogram(n_fft = self.n_fft)
        self.invs = torchaudio.transforms.GriffinLim(n_fft = self.n_fft)

        self.melmat = torchaudio.functional.create_fb_matrix(n_freqs = self.n_fft // 2 + 1, 
                                                             f_min = 0, 
                                                             f_max = self.sr / 2, 
                                                             n_mels = self.n_mels, 
                                                             sample_rate = self.sr)

        self.smoothing = smoothing
        self.smoother = torch.hann_window(2 * self.smoothing + 1)
        self.smoother = self.smoother / torch.sum(self.smoother).view(1, -1)
        
    def __getitem__(self, idx):
        noise = torch.rand(self.num_channel, self.noise_len + self.n_fft)
        db_rand = torch.rand(self.num_channel, self.n_mels) * (self.max_db - self.min_db) + self.min_db
        db_rand = dsp.convolve(db_rand, self.smoother)[:, self.smoothing:-self.smoothing]
        gain_rand = 10 ** (db_rand / 20)
        mel_gain = torch.sum(self.melmat * gain_rand.unsqueeze(-2), -1)
        rdnoise = self.invs(self.spec(noise) * mel_gain.unsqueeze(-1))
        return noise[:, :self.noise_len].squeeze(), rdnoise[:, :self.noise_len].squeeze()

    def __len__(self):
        return self.num_channel * self.num_batch

