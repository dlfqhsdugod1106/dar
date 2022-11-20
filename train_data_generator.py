import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader

import numpy as np
from tqdm import tqdm
import os

from train_reverb.dataset.rir_dataset.pyroom_generator_2 import get_random_ir

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type = str, default = '/SSD/pyroom_data/')
parser.add_argument('--sr', type = int, default = 48000, help = 'sr to render')
parser.add_argument('--num_data', type = int, default = 100, help = 'num of ir to generate')
parser.add_argument('--maximum_order', type = int, default = 120, help = 'maximum order of ISM to render')
parser.add_argument('--render_sec', type = float, default = 5., help = 'maximum RT60 to accurately compute')
parser.add_argument('--num_workers', type = int, default = 16, help = 'parallel workers to compute ISM')
args = parser.parse_args()

class PyRoom_Generator(Dataset):
    def __init__(self, save_dir, num_data = 100, sr = 48000, maximum_order = 120, render_sec = 2.5):
        super().__init__()

        self.save_dir = save_dir
        self.num_data = num_data
        self.sr = sr
        self.maximum_order = maximum_order
        self.render_sec = render_sec

    def __getitem__(self, idx):
        ir = get_random_ir(sr = self.sr, maximum_order = self.maximum_order, render_sec = self.render_sec)
        np.save(args.save_dir + str(100 + idx // 1000).zfill(3) + '/' + str(idx % 1000).zfill(3) + '.npy', ir)
        return torch.ones(1)

    def __len__(self):
        return self.num_data

generator = PyRoom_Generator(save_dir = args.save_dir, 
        num_data = args.num_data, 
        sr = args.sr, 
        maximum_order = args.maximum_order, 
        render_sec = args.render_sec)

loader = DataLoader(generator, batch_size = 1, shuffle = False, num_workers = args.num_workers, 
        worker_init_fn = lambda _: np.random.seed())

num_subdir = args.num_data // 1000
for i in range(num_subdir): 
    os.makedirs(args.save_dir + str(100 + i).zfill(3), exist_ok = True)

for idx, _ in enumerate(tqdm(loader)):
    pass
