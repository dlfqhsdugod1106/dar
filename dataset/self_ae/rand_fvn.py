import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from components import fvn as FVN
from components import core_complex as COMPLEX

class RandFVN(Dataset):
    def __init__(self, num_data, fvn, coeff_length):
        super().__init__()
        self.num_data = num_data
        self.coeff_length = coeff_length        

        self.z = {}
       
        ds_coeff = self.coeff_length // 4
        ups = nn.Upsample(self.coeff_length, mode = 'linear')
        
        self.amp = ups(torch.ones(self.num_data, fvn.seg_num, ds_coeff, device = fvn.device) * .9 \
            + torch.rand(self.num_data, fvn.seg_num, ds_coeff, device = fvn.device) * .1)
        self.pha = ups(torch.rand(self.num_data, fvn.seg_num, ds_coeff, device = fvn.device) * 2 * np.pi - np.pi) 
        
        self.z['H'] = COMPLEX.POLAR_TO_CART(torch.stack([self.amp, self.pha], -1))
         
        self.fvn_ir = fvn.forward(self.z)
 
    def __getitem__(self, idx):
        return self.z['H'][idx], self.fvn_ir[idx]
     
    def __len__(self):
        return self.num_data

