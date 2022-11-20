"""
2020_04_03 - 2020_05_12
BUTREVDB RIR Dataset.
TODO: Metadata Analysis, RIR Stat Anaylsis.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../") 

from metric.ir_profile import ir_profile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import glob

class BUTREVDB(Dataset):
    def __init__(self, 
                 directory = "/SSD/BUTREVDB/RIR_only",
                 param_extract = False): 
        
        super().__init__()

        self.ir_directory_list = glob.glob(directory + '/*/*/*/*/RIR/*.wav')
        self.num_ir = len(self.ir_directory_list)
        
        self.param_extract = param_extract

    def __getitem__(self, idx):
        data, sr = sf.read(self.ir_directory_list[idx])

        if len(data.shape) != 1:
            print("SIBAL!")

        if self.param_extract:
            with torch.no_grad():
                profile = ir_profile(torch.tensor(data).view(1, -1), 
                        ir_ms = data.shape[0] * 1e3 / sr)

                profile = torch.stack(list(profile.values()), -2).squeeze()
                return profile
        else:
            return data
    
    def __len__(self):
        return self.num_ir
