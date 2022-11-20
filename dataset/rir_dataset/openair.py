"""
2020_05_01 - 2020_05_12
OpenAIR RIR Dataset.
TODO: Metadata Analysis, RIR Stat Anaylsis.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from metric.ir_profile import ir_profile

import numpy as np
import soundfile as sf
import torch

from scipy import signal

from torch.utils.data import Dataset
import glob


class OpenAIR(Dataset):
    def __init__(self, 
                 directory = '/SSD/RIR/OpenAIR',
                 param_extract = False, 
                 out_sr = None,
                 out_len = None):
        
        super().__init__()
        
        self.ir_directory_list = []
        self.place_directory_list = glob.glob(directory + '/*')

        mic_channel = ['mono', 'stereo', 'b-format', 'surround-5-1']
        for place in self.place_directory_list:
            mic_list = [glob.glob(place + '/' + mic) for mic in mic_channel]
            for mic_dir in mic_list:
                if len(mic_dir) != 0:
                    ir_directory = glob.glob(mic_dir[0] + '/*.wav')
                    self.ir_directory_list += ir_directory
                    break
                    
        self.param_extract = param_extract
        self.out_sr = out_sr
        self.out_len = out_len

    def __getitem__(self, idx):
        data, sr = sf.read(self.ir_directory_list[idx])
        if len(data.shape) == 2:
            data = data[..., 0]
        if self.param_extract:
            with torch.no_grad():
                #if data.shape[0] > sr * 5: # LOCK MAX IR LEN TO 5 SEC, AT LEAST FOR NOW
                #    data = data[:sr * 5]
                sig_len = len(data) / sr
                data = signal.resample(data, int(16000 * sig_len))
                profile = ir_profile(torch.tensor(data).view(1, -1), 
                        ir_ms = sig_len * 1e3)
                profile = torch.stack(list(profile.values()), -2).squeeze()
                return profile

        if self.out_len is not None:
            if len(data) > sr * self.out_len:
                data = data[:sr * self.out_len]
            else:
                data = np.pad(data, (0, sr * self.out_len - len(data)))

        if self.out_sr is not None:
            sig_len = len(data) / sr
            data = signal.resample(data, int(self.out_sr * sig_len))
        #metadata = {}
        #metadata['dir'] = self.ir_directory_list[idx]
        return data#, metadata

    def __len__(self):
        return len(self.ir_directory_list)
