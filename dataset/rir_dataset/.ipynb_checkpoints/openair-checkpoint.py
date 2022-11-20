"""
2020_05_01 - 2020_05_12
OpenAIR RIR Dataset.
TODO: Metadata Analysis, RIR Stat Anaylsis.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import glob

class OpenAIR(Dataset):
    def __init__(self, 
                 directory, 
                 sr = 48000):
        
        super().__init__()
        
        self.sr = sr
        
        self.ir_directory_list = []
        self.place_directory_list = glob.glob(directory + '/*')
        
        
        mic_channel = ['mono', 'stereo', 'b-format', 'surround-5-1']
        for place in self.place_directory_list:
            mic_list = [glob.glob(place + '/' + mic) for mic in mic_channel]
            for mic_dir in mic_list:
                if len(mic_dir) != 0:
                    ir_directory = glob.glob(mic_dir[0] + '/*')
                    self.ir_directory_list += ir_directory
                    break
                    
    def __getitem__(self, idx):
        data, _ = librosa.load(self.ir_directory_list[idx], sr = self.sr, mono = True)
        metadata = {}
        metadata['dir'] = self.ir_directory_list[idx]
        return data, metadata

    def __len__(self):
        return len(self.ir_directory_list)