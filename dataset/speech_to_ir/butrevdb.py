"""
2020_06_02
BUTREVDB transmitted speech & RIR dataset.
"""

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import glob
import random 

class BUTREVDB_T(Dataset):
    def __init__(self,
            speech_length = 32000,
            tspeech_directory = '/SSD/BUTREVDB/Transmitted',
            dryspeech_directory = '/SSD/LibriSpeech/test-clean',
            rir_directory = '/SSD/BUTREVDB/RIR_only',
            prototype = False):
		
        super().__init__()
        
        self.speech_length = speech_length

        self.tspeech_directory = tspeech_directory
        self.tspeech_directory_list = glob.glob(self.tspeech_directory
            + '/*/*/*/*/english/LibriSpeech/test-clean/*/*/*.v00.wav')
		
        self.dryspeech_directory = dryspeech_directory

        self.num_tspeech = len(self.tspeech_directory_list)
        
        self.setups = [tsdir[len(self.tspeech_directory):len(self.tspeech_directory) + 43] 
            for tsdir in self.tspeech_directory_list]
        self.setups = list(set(self.setups))

        self.rir = {}
        for setup in self.setups:
            self.rir[setup], _ = sf.read(rir_directory + setup 
                + '/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav') 

        self.prototype = prototype

    def __getitem__(self, idx):
        directory = self.tspeech_directory_list[idx]
        tspeech, _ = sf.read(directory)
        
        setup = directory[len(self.tspeech_directory):len(self.tspeech_directory) + 43]
        dry_setup = directory[len(self.tspeech_directory) + 75:-8]
        dryspeech, _ = sf.read(self.dryspeech_directory + '/' + dry_setup + '.flac')

        margin = len(dryspeech) - self.speech_length
        
        if margin < 0:
            dryspeech = np.pad(dryspeech, (0, -margin))
        else:
            t0 = random.randint(0, margin)
            dryspeech = dryspeech[t0:t0 + self.speech_length]
            tspeech = tspeech[t0:]
        
        if len(tspeech) < self.speech_length:
            tspeech = np.pad(tspeech, (0, self.speech_length - len(tspeech)))
        else:
            tspeech = tspeech[:self.speech_length]

        rir = self.rir[setup]
        
        return tspeech, rir, setup, dryspeech, dry_setup

    def __len__(self):
        if self.prototype: return 3000
        else: return self.num_tspeech
