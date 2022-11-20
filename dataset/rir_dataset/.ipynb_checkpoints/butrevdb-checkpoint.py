"""
2020_04_03 - 2020_05_12
BUTREVDB RIR Dataset.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import glob

class BUTREVDB(Dataset):
    def __init__(self, 
                 directory, 
                 sr = 16000):
        
        super().__init__()
        self.ir_directory_list = glob.glob(directory + '/*/*/*/*/RIR/*')
        self.sr = sr
        self.num_ir = len(self.ir_directory_list)
        
    def __getitem__(self, idx):
        data, _ = librosa.load(self.ir_directory_list[idx], sr = self.sr)
        return data
    
    def __len__(self):
        return self.num_ir