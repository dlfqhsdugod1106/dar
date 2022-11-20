"""
2020_11_25
Speech related
"""

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import random
import numpy as np
import soundfile as sf
from scipy import signal

import glob

def get_timit(dataset_dir = '/SSD/TIMIT/timit/TIMIT',
              train_ratio = .8,
              boost_ratio = 10):

    train_dir = dataset_dir + '/TRAIN'
    train_spk = glob.glob(train_dir + '/*/*')
    random.shuffle(train_spk)

    split_idx = int(len(train_spk) * train_ratio)
    train_spk_div = train_spk[:split_idx]
    vali_spk_div = train_spk[split_idx:]

    train_list = []
    for spk in train_spk_div:
        train_list += glob.glob(spk + '/*.WAV')

    vali_list = []
    for spk in vali_spk_div:
        vali_list += glob.glob(spk + '/*.WAV')

    test_dir = dataset_dir + '/TEST'
    test_list = glob.glob(test_dir + '/*/*/*.WAV')

    train_list = train_list * boost_ratio

    print("TIMIT:", len(train_list), len(vali_list), len(test_list))

    return train_list, vali_list, test_list


def get_libri(train_dir = '/SSD/LibriSpeech/train-clean-360',
              test_dir = '/SSD/LibriSpeech/test-clean',
              train_ratio = 0.95):

    train_spk = glob.glob(train_dir + '/*')
    random.shuffle(train_spk)

    split_idx = int(len(train_spk) * train_ratio)
    train_spk_div = train_spk[:split_idx]
    vali_spk_div = train_spk[split_idx:]

    train_list = []
    for spk in train_spk_div:
        train_list += glob.glob(spk + '/*/*.flac')

    vali_list = []
    for spk in vali_spk_div:
        vali_list += glob.glob(spk + '/*/*.flac')

    test_list = glob.glob(test_dir + '/*/*/*.flac')

    print("LIBRI:", len(train_list), len(vali_list), len(test_list))

    return train_list, vali_list, test_list




