"""
2020_09_25
Parent class for IR data handling
"""

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import numpy as np
import soundfile as sf
from scipy import signal

import torch
from torch.utils.data import Dataset, ConcatDataset

import glob
import random

from metric.ir_profile import ir_profile
from dataset.rir_dataset.augment_2 import *


def Generate_Dataset():
    train_speech, vali_speech, test_speech = get_speech_lists()

    for speech in train_speech + vali_speech + test_speech:
        if speech[-3:] != 'wav' and speech[-4:] != 'flac' and speech[-3:] != 'WAV':
            print(speech)

    train_alti, vali_alti, test_alti = ALTI_SPLIT([train_speech, vali_speech, test_speech])
    train_ace, vali_ace, test_ace = ACE_SPLIT([train_speech, vali_speech, test_speech])
    train_but, vali_but, test_but = BUTREVDB_SPLIT([train_speech, vali_speech, test_speech])
    train_open, vali_open, test_open = OPENAIR_SPLIT([train_speech, vali_speech, test_speech])
    train_gh = GREATHALL(train_speech)
    train_mardy = MARDY(train_speech)
    #train_mega = MEGAVERB(train_speech)
    train_pror = PROR(train_speech)
    #train_rc24 = RC24(train_speech)
    #train_rc48 = RC48(train_speech)
    #train_simir = SIMIR(train_speech)

    print('alti', len(train_alti), len(vali_alti), len(test_alti))
    print('ace', len(train_ace), len(vali_ace), len(test_ace))
    print('but', len(train_but), len(vali_but), len(test_but))
    print('gh', len(train_gh), '0', '0')
    print('mard', len(train_mardy), '0', '0')
    print('open', len(train_open), len(vali_open), len(test_open))
    #print('mega', len(train_mega), '0', '0')
    print('pror', len(train_pror), '0', '0')
    #print('rc24', len(train_rc24), '0', '0')
    #print('rc48', len(train_rc48), '0', '0')
    #print('sim', len(train_simir), '0', '0')

    train_list = [train_alti,  
                  train_ace, 
                  train_but, 
                  train_open,
                  train_gh,
                  train_mardy,]
                  #train_mega,
                  #train_pror,]
                  #train_rc24,
                  #train_rc48,
                  #train_simir]

    train_ir = ConcatDataset(train_list)

    vali_list = [vali_alti,
                 vali_ace,
                 vali_but,
                 vali_open]

    vali_ir = ConcatDataset(vali_list)

    test_list = [test_alti,
                 test_ace,
                 test_but,
                 test_open]

    test_ir = ConcatDataset(test_list)

    print(len(train_ir), len(vali_ir), len(test_ir))

    return train_ir, vali_ir, test_ir


def get_speech_lists():
    train_timit, vali_timit, test_timit = get_timit()
    #train_libri, vali_libri, test_libri = get_libri()
    train_vctk, vali_vctk, test_vctk = get_vctk()

    train_speech = train_vctk + train_timit
    vali_speech = vali_vctk + vali_timit
    test_speech = test_vctk + test_timit

    random.shuffle(train_speech)
    random.shuffle(vali_speech)
    random.shuffle(test_speech)

    print(len(train_speech), len(vali_speech), len(test_speech))

    return train_speech, vali_speech, test_speech


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


def get_vctk(dataset_dir = '/SSD/VCTK/vctk',
             train_ratio = 0.7,
             vali_ratio = 0.25):

    spk_list = glob.glob(dataset_dir + '/wav48_silence_trimmed/*')

    split_idx_1 = int(len(spk_list) * train_ratio)
    split_idx_2 = int(len(spk_list) * (train_ratio + vali_ratio))

    train_spk_div = spk_list[:split_idx_1]
    vali_spk_div = spk_list[split_idx_1:split_idx_2]
    test_spk_div = spk_list[split_idx_2:]

    train_list = []
    for spk in train_spk_div:
        train_list += glob.glob(spk + '/*.flac')

    vali_list = []
    for spk in vali_spk_div:
        vali_list += glob.glob(spk + '/*.flac')

    test_list = []
    for spk in test_spk_div:
        test_list += glob.glob(spk + '/*.flac')

    print("VCTK :", len(train_list), len(vali_list), len(test_list))

    return train_list, vali_list, test_list



def cal_rms(amp, eps = 1e-7):
    return np.sqrt(np.mean(np.square(amp), axis=-1) + eps)

def rms_normalize(wav, ref_dBFS=-23.0, eps = 1e-7):
    rms = cal_rms(wav)
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    gain = ref_linear / (rms + eps)
    wav = gain * wav
    return wav    




class IRDataset(Dataset):
    def __init__(self,
                 ir_directory_list,
                 speech_directory_list,
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1, 
                 speech_len = 3,
                 sample_ratio = 1.,
                 aug = True,
                 aug_const = 1,
                 **aug_kwargs):

        super().__init__()
        
        self.ir_directory_list = ir_directory_list 
        self.speech_directory_list = speech_directory_list
        self.speech_dataset_len = len(self.speech_directory_list)
        self.param_extract = param_extract 
        self.out_sr = out_sr
        self.out_len = out_len
        self.speech_len = speech_len
        self.sample_ratio = sample_ratio
        self.aug = aug
        self.aug_const = aug_const
        self.aug_kwargs = aug_kwargs

    def __getitem__(self, idx):

        """ IR """
        if self.sample_ratio != 1.:
            idx = random.randint(0, len(self.ir_directory_list) - 1)

        data, sr = sf.read(self.ir_directory_list[idx % len(self.ir_directory_list)])

        if len(data.shape) == 2:
            data = data[..., 0]
        
        if self.out_len is not None:
            if len(data) > sr * self.out_len:
                data = data[:sr * self.out_len]
            else:
                data = np.pad(data, (0, sr * self.out_len - len(data)))

        if self.out_sr is not None:
            if sr != self.out_sr:
                sig_len = len(data) / sr
                data = signal.resample(data, int(self.out_sr * sig_len))

        onset = get_onset(data)

        if onset > 10:
            randint = random.randrange(0, 10)
            data = np.concatenate([data[onset - randint:], np.zeros(onset - randint)])

        data = data / np.sqrt(np.sum(data ** 2 + 1e-7))  ### NEW!!

        #if self.aug:
        #    data = augment(data, **self.aug_kwargs)


        """ IR ANALYSIS MODE """
        if self.param_extract:
            with torch.no_grad():
                sig_len = len(data) / sr
                data = signal.resample(data, int(16000 * sig_len))
                profile = ir_profile(torch.tensor(data).view(1, -1), ir_ms = sig_len * 1e3)
                profile = torch.stack(list(profile.values()), -2).squeeze()
                return profile, self.ir_directory_list[idx]

        """ SPEECH """
        speech_idx = random.randint(0, self.speech_dataset_len - 1)
        speech, speech_sr = sf.read(self.speech_directory_list[speech_idx])
        
        if len(speech.shape) == 2:
            speech = speech[..., 0]
        

        dry_len = self.speech_len + self.out_len
        margin = len(speech) - dry_len * speech_sr

        if margin < 0:
            dryspeech_cut = np.pad(speech, (0, -margin))
        else:
            t0 = random.randint(0, margin)
            dryspeech_cut = speech[t0:t0 + dry_len * speech_sr]

        if speech_sr != self.out_sr:
            dryspeech_cut = signal.resample(dryspeech_cut, int(self.out_sr * dry_len))

        dryspeech_cut = speech_augment(dryspeech_cut) ## !!
        #dryspeech_cut = rms_normalize(dryspeech_cut)

        tspeech = signal.convolve(dryspeech_cut, data)[:int(self.out_sr * dry_len)]
        dryspeech_cut = dryspeech_cut[len(data):]
        tspeech = tspeech[len(data):]

        return data, tspeech, dryspeech_cut

    def __len__(self):
        return int(len(self.ir_directory_list) * self.aug_const * self.sample_ratio)


def ALTI_SPLIT(speech_lists, directory = '/SSD/RIR/altiverb-ir', split_ratio = (.6, .2), aug_const = 3):
    total_directory_list = glob.glob(directory + '/*/*/*.wav')

    place_list = [directory.split(' - ')[0] for directory in total_directory_list]
    place_list = list(set(place_list))
    place_list = sorted(place_list)

    random.shuffle(place_list)

    num_train = int(len(place_list) * split_ratio[0])
    num_vali = int(len(place_list) * split_ratio[1])

    train_place_list = place_list[:num_train]
    vali_place_list = place_list[num_train:num_train + num_vali]
    test_place_list = place_list[num_train + num_vali:] 

    train_directory_list = []
    for train_place in train_place_list:
        train_directory_list += glob.glob(train_place + '*.wav')

    vali_directory_list = []
    for vali_place in vali_place_list:
        vali_directory_list += glob.glob(vali_place + '*.wav')

    test_directory_list = []
    for test_place in test_place_list:
        test_directory_list += glob.glob(test_place + '*.wav')


    train_alti = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = True)
    vali_alti = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False)
    test_alti = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False)

    return train_alti, vali_alti, test_alti


class ALTIVERB(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = '/SSD/RIR/altiverb-ir',
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*/*/*.wav')

        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def ACE_SPLIT(speech_lists, directory = '/SSD/RIR/ACE', split_ratio = (.6, .2), aug_const = 50):
    total_directory_list = glob.glob(directory + '/*.wav')

    place_list = [directory.split('_')[0] + '_' + directory.split('_')[1] 
                  for directory in total_directory_list]
    place_list = list(set(place_list))

    random.shuffle(place_list)

    num_train = int(len(place_list) * split_ratio[0])
    num_vali = int(len(place_list) * split_ratio[1])

    train_place_list = place_list[:num_train]
    vali_place_list = place_list[num_train:num_train + num_vali]
    test_place_list = place_list[num_train + num_vali:] 

    train_directory_list = []
    for train_place in train_place_list:
        train_directory_list += glob.glob(train_place + '*.wav')

    vali_directory_list = []
    for vali_place in vali_place_list:
        vali_directory_list += glob.glob(vali_place + '*.wav')

    test_directory_list = []
    for test_place in test_place_list:
        test_directory_list += glob.glob(test_place + '*.wav')

    train_ace = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = True)
    vali_ace = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False)
    test_ace = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False)

    return train_ace, vali_ace, test_ace


class ACE(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = '/SSD/RIR/ACE',
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*.wav')

        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def BUTREVDB_SPLIT(speech_lists, directory = '/SSD/BUTREVDB/RIR_only', split_ratio = (.6, .2)):
    place_list = glob.glob(directory + '/VUT*')
    random.shuffle(place_list)

    num_train = int(len(place_list) * split_ratio[0])
    num_vali = int(len(place_list) * split_ratio[1])

    train_place_list = place_list[:num_train]
    vali_place_list = place_list[num_train:num_train + num_vali]
    test_place_list = place_list[num_train + num_vali:] 

    train_directory_list = []
    for train_place in train_place_list:
        train_directory_list += glob.glob(train_place + '/*/*/*/RIR/*.wav')

    vali_directory_list = []
    for vali_place in vali_place_list:
        vali_directory_list += glob.glob(vali_place + '/*/*/*/RIR/*.wav')

    test_directory_list = []
    for test_place in test_place_list:
        test_directory_list += glob.glob(test_place + '/*/*/*/RIR/*.wav')

    train_but = IRDataset(train_directory_list, speech_lists[0], aug = True)
    vali_but = IRDataset(vali_directory_list, speech_lists[1], aug = False)
    test_but = IRDataset(test_directory_list, speech_lists[2], aug = False)

    return train_but, vali_but, test_but

class BUTREVDB(IRDataset):
    def __init__(self,
                 directory = "/SSD/BUTREVDB/RIR_only",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*/*/*/*/RIR/*.wav')
        
        super().__init__(self.ir_directory_list, param_extract, out_sr, out_len)


class GREATHALL(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/RIR/greathall",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


class MARDY(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/RIR/MARDY",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1,
                 aug_const = 2):

        self.ir_directory_list = glob.glob(directory + '/*.wav')
        
        super().__init__(self.ir_directory_list, 
                         speech_directory_list, 
                         param_extract, 
                         out_sr, 
                         out_len, 
                         aug_const = aug_const)


class MEGAVERB(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/ARIR/megaverb",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def OPENAIR_SPLIT(speech_lists, directory = '/SSD/RIR/OpenAIR', split_ratio = (.6, .2), aug_const = 10):
    place_list = glob.glob(directory + '/*')
    random.shuffle(place_list)

    num_train = int(len(place_list) * split_ratio[0])
    num_vali = int(len(place_list) * split_ratio[1])

    train_place_list = place_list[:num_train]
    vali_place_list = place_list[num_train:num_train + num_vali]
    test_place_list = place_list[num_train + num_vali:] 

    mic_channel = ['mono', 'stereo', 'b-format', 'surround-5-1']

    train_directory_list = []
    for train_place in train_place_list:
        mic_list = [glob.glob(train_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob.glob(mic_dir[0] + '/*.wav')
                train_directory_list += ir_directory
                break

    vali_directory_list = []
    for vali_place in vali_place_list:
        mic_list = [glob.glob(vali_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob.glob(mic_dir[0] + '/*.wav')
                vali_directory_list += ir_directory
                break

    test_directory_list = []
    for test_place in test_place_list:
        mic_list = [glob.glob(test_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob.glob(mic_dir[0] + '/*.wav')
                test_directory_list += ir_directory
                break

    train_openair = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = True)
    vali_openair = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False)
    test_openair = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False)

    return train_openair, vali_openair, test_openair


class OPENAIR(IRDataset):
    def __init__(self,
                 directory = "/SSD/RIR/OpenAIR",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

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
        
        super().__init__(self.ir_directory_list, param_extract, out_sr, out_len)


class PROR(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/ARIR/pro-r",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


class RC24(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/ARIR/rc24",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1,
                 sample_ratio = .5):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, 
                         speech_directory_list, 
                         param_extract, 
                         out_sr, 
                         out_len, 
                         sample_ratio = sample_ratio)


class RC48(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/ARIR/rc48",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1,
                 sample_ratio = .5):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, 
                         speech_directory_list, 
                         param_extract, 
                         out_sr, 
                         out_len, 
                         sample_ratio = sample_ratio)


class SIMIR(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/RIR/RIRS_NOISES/",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 1,
                 sample_ratio = .01):

        self.ir_directory_list = glob.glob(directory + 'simulated_rirs/*/*/*.wav')
        
        super().__init__(self.ir_directory_list, 
                         speech_directory_list, 
                         param_extract, 
                         out_sr, 
                         out_len, 
                         sample_ratio = sample_ratio)
