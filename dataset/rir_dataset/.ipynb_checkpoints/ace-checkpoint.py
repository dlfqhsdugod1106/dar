"""
2020_05_01 - 2020_05_12
ACE Challenge RIR Dataset.
TODO: Metadata Analysis, RIR Stat Anaylsis.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import glob

class ACE(Dataset):
    def __init__(self, 
                 directory, 
                 sr = 48000,
                 all_channel = True):
        
        super().__init__()
        
        self.ir_directory_list = glob.glob(directory + '/*/*/*RIR.wav')
        self.num_ir = len(self.ir_directory_list)
        self.sr = sr
        self.all_channel = all_channel
        data, _ = librosa.load(self.ir_directory_list[idx], sr = self.sr, mono = False)
        self.num_channel = data.shape[0]
        self.sr = sr
        
    def __getitem__(self, idx):
        if self.all_channel:
            data, _ = librosa.load(self.ir_directory_list[idx // self.num_channel], sr = self.sr, mono = False)
            return data[idx % self.num_channel]
        else:
            data, _ = librosa.load(self.ir_directory_list[idx], sr = self.sr, mono = True)
            return data
    
    def __len__(self):
        if self.all_channel:
            return self.num_ir * self.num_channel
        else:
            return self.num_ir