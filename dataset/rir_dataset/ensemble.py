"""
2020_05_15
RIR Dataset Ensemble.
TODO: Metadata Analysis, RIR Stat Anaylsis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from dataset.rir_dataset import ace, butrevdb, greathall, mardy, openair 

class RIREnsemble(Dataset):
    def __init__(self,
                 dir_list,): 
        
        super().__init__()
        
        self.dir_list = dir_list
        self.dataset_list = []
        if self.dir_list['ace'] is not None:
            self.dataset_list.append(ace.ACE(self.dir_list['ace'],))
        if self.dir_list['butrevdb'] is not None:
            self.dataset_list.append(butrevdb.BUTREVDB(self.dir_list['butrevdb'],))
        if self.dir_list['greathall'] is not None:
            self.dataset_list.append(greathall.GreatHall(self.dir_list['greathall'],))
        if self.dir_list['mardy'] is not None:
            self.dataset_list.append(mardy.MARDY(self.dir_list['mardy'],))
        if self.dir_list['openair'] is not None:
            self.dataset_list.append(openair.OpenAIR(self.dir_list['openair'],))

    def __getitem__(self, idx):
        for i in range(len(self.dataset_list)):
            if idx < len(self.dataset_list[i]):
                data = (self.dataset_list[i])[idx]
            else:
                idx -= len(self.dataset_list[i])
        return data
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.dataset_list])
