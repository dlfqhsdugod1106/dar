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
import librosa

from metric.ir_profile import ir_profile
from dataset.rir_dataset.augment_3 import *
from dataset.rir_dataset.pyroom_generator_2 import *
from dataset.speech_to_ir.speech import *

def Generate_Dataset(ir_len = 2, noiseir_ratio = .3, sr = 16000, speech_len = 3, ism_max_order = 50, rdeq = True, rdeq_range = (1, 3)):
    train_speech, vali_speech, test_speech = get_speech_lists()

    vali_alti, test_alti_1, test_alti_2 = ALTI_SPLIT([vali_speech, test_speech, test_speech], 
                                                      split_ratio = (.5, .2), 
                                                      out_len = ir_len, 
                                                      sr = sr)

    vali_ace, test_ace_1, test_ace_2 = ACE_SPLIT([vali_speech, test_speech, test_speech], 
                                                  split_ratio = (.5, .2), 
                                                  out_len = ir_len,
                                                  sr = sr)

    vali_open, test_open_1, test_open_2 = OPENAIR_SPLIT([vali_speech, test_speech, test_speech], 
                                                         split_ratio = (.5, .2), 
                                                         out_len = ir_len,
                                                         sr = sr)

    vali_list = [vali_alti, vali_ace, vali_open]
    test_list = [test_alti_1, test_alti_2, test_ace_1, test_ace_2, test_open_1, test_open_2]

    train_ir = PyRoom3_saved(train_speech, sr = sr, ir_len = ir_len, noiseir_ratio = noiseir_ratio, speech_len = speech_len, rdeq = rdeq, rdeq_range = rdeq_range)
    vali_ir = ConcatDataset(vali_list)
    #test_ir = ConcatDataset(test_list)

    #print("TRAIN", len(train_ir), "VALIDATION", len(vali_ir), "TEST", len(test_ir))
    return train_ir, vali_ir, test_list

def get_speech_lists():
    train_timit, vali_timit, test_timit = get_timit()
    train_vctk, vali_vctk, test_vctk = get_vctk()

    train_speech = train_vctk + train_timit
    vali_speech = vali_vctk + vali_timit
    test_speech = test_vctk + test_timit

    random.shuffle(train_speech)
    random.shuffle(vali_speech)
    random.shuffle(test_speech)

    print(len(train_speech), len(vali_speech), len(test_speech))

    return train_speech, vali_speech, test_speech


def cal_rms(amp, eps = 1e-7):
    return np.sqrt(np.mean(np.square(amp), axis=-1) + eps)

def rms_normalize(wav, ref_dBFS=-23.0, eps = 1e-7):
    rms = cal_rms(wav)
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    gain = ref_linear / (rms + eps)
    wav = gain * wav
    return wav    


                 
class PyRoom3_saved(Dataset):
    def __init__(self, 
                 speech_directory_list,
                 pyroom_directory = '/HDD1/pyroom_data/',
                 sr = 16000,
                 ir_len = 2.5,
                 speech_len = 3,
                 noiseir_ratio = 0.3,
                 rdeq = True,
                 rdeq_range = (2, 6)):

        super().__init__()

        self.speech_directory_list = speech_directory_list
        self.speech_dataset_len = len(self.speech_directory_list)

        self.ir_directory = sorted(glob.glob(pyroom_directory + '*/*.npy'))

        self.sr = sr
        self.ir_len = ir_len 
        self.speech_len = speech_len
        self.noiseir_ratio = noiseir_ratio
        self.rdeq = rdeq
        self.rdeq_range = rdeq_range


    def __getitem__(self, idx):
        if np.random.uniform() > self.noiseir_ratio:
            ir = np.load(self.ir_directory[idx])

            onset = get_onset(ir)
            if onset > 10:
                randint = random.randrange(0, 10)
                ir = np.concatenate([ir[onset - randint:], np.zeros(onset - randint)])

            da_window, lr_window = splitter_window(0, num_sample = len(ir), sr = 48000)
            da_ir, lr_ir = ir * da_window, ir * lr_window
            da_strength = np.random.uniform(-12, 3)
            da_strength = 10 ** (da_strength / 20)
            ir = da_ir * da_strength + lr_ir

            if self.sr != 48000:
                ir = librosa.resample(ir, 48000, self.sr)

            if len(ir) < self.sr * self.ir_len:
                ir = np.pad(ir, (0, int(self.sr * self.ir_len) - len(ir)))
        else:
            ir = rdnoise_ir(ir_len = int(self.sr * self.ir_len))

        if self.rdeq:
            ir = rdeq2(ir)
        ir = ir / np.sqrt(np.sum(ir ** 2) + 1e-12)


        """ SPEECH """
        speech_idx = random.randint(0, self.speech_dataset_len - 1)
        speech, speech_sr = sf.read(self.speech_directory_list[speech_idx])
        
        if len(speech.shape) == 2:
            speech = speech[..., 0]

        speech = np.concatenate([speech, speech], 0)

        dry_len = self.speech_len + self.ir_len
        margin = len(speech) - int(dry_len * speech_sr)

        if margin < 0:
            dryspeech_cut = np.pad(speech, (0, -margin))
        else:
            t0 = random.randint(0, margin)
            dryspeech_cut = speech[t0:t0 + int(dry_len * speech_sr)]

        if speech_sr != self.sr:
            dryspeech_cut = librosa.resample(dryspeech_cut, speech_sr, self.sr)

        #dryspeech_cut = speech_augment(dryspeech_cut)
        dryspeech_cut = rms_normalize(dryspeech_cut)

        tspeech = signal.convolve(dryspeech_cut, ir)[:int(self.sr * dry_len)]

        if len(ir) > self.sr * self.ir_len:
            ir = ir[:int(self.sr * self.ir_len)]

        dryspeech_cut = dryspeech_cut[len(ir):]
        tspeech = tspeech[len(ir):]

        return ir, tspeech, dryspeech_cut

    def __len__(self):
        return len(self.ir_directory)


class IRDataset(Dataset):
    def __init__(self,
                 ir_directory_list,
                 speech_directory_list,
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2,  # 1
                 speech_len = 3,
                 sample_ratio = 1.,
                 aug = True,
                 get_speech_idx = False,
                 aug_const = 1,
                 **aug_kwargs):

        super().__init__()
        
        self.ir_directory_list = ir_directory_list 
        random.shuffle(self.ir_directory_list)
        self.speech_directory_list = speech_directory_list
        self.speech_dataset_len = len(self.speech_directory_list)
        self.param_extract = param_extract 
        self.out_sr = out_sr
        self.out_len = out_len
        self.speech_len = speech_len
        self.sample_ratio = sample_ratio
        self.get_speech_idx = get_speech_idx
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
        
        onset = get_onset(data)

        if onset > 10:
            randint = random.randrange(0, 10)
            data = np.concatenate([data[onset - randint:], np.zeros(onset - randint)])

        if len(data) < sr * self.out_len:
            data = np.pad(data, (0, int(sr * self.out_len) - len(data)))

        if self.out_sr is not None:
            if sr != self.out_sr:
                sig_len = len(data) / sr
                data = librosa.resample(data, sr, self.out_sr)

        data = data / np.sqrt(np.sum(data ** 2) + 1e-12)  ### NEW!!

        #if self.aug:
        #    data = augment(data, **self.aug_kwargs)


        """ IR ANALYSIS MODE """
        #if self.param_extract:
        #    with torch.no_grad():
        #        sig_len = len(data) / sr
        #        data = signal.resample(data, int(16000 * sig_len))
        #        profile = ir_profile(torch.tensor(data).view(1, -1), ir_ms = sig_len * 1e3)
        #        profile = torch.stack(list(profile.values()), -2).squeeze()
        #        return profile, self.ir_directory_list[idx]

        """ SPEECH """
        speech_idx = random.randint(0, self.speech_dataset_len - 1)
        speech, speech_sr = sf.read(self.speech_directory_list[speech_idx])
        
        if len(speech.shape) == 2:
            speech = speech[..., 0]
        
        speech = np.concatenate([speech, speech], 0)

        dry_len = self.speech_len + self.out_len
        margin = len(speech) - int(dry_len * speech_sr)

        if margin < 0:
            dryspeech_cut = np.pad(speech, (0, -margin))
        else:
            t0 = random.randint(0, margin)
            dryspeech_cut = speech[t0:t0 + int(dry_len * speech_sr)]

        if speech_sr != self.out_sr:
            dryspeech_cut = librosa.resample(dryspeech_cut, speech_sr, self.out_sr)

        #dryspeech_cut = speech_augment(dryspeech_cut) ## !!
        dryspeech_cut = rms_normalize(dryspeech_cut)

        tspeech = signal.convolve(dryspeech_cut, data)[:int(self.out_sr * dry_len)]

        data = data[:int(self.out_sr * self.out_len)]

        dryspeech_cut = dryspeech_cut[len(data):]
        tspeech = tspeech[len(data):]

        if self.get_speech_idx:
            return data, tspeech, dryspeech_cut, speech_idx
        else:
            return data, tspeech, dryspeech_cut

    def __len__(self):
        return int(len(self.ir_directory_list) * self.aug_const * self.sample_ratio)

    def get_full_data(self, speech_idx, rir_idx):
        speech, speech_sr = sf.read(self.speech_directory_list[speech_idx])
        if len(speech.shape) == 2:
            speech = speech[..., 0]
        if speech_sr != self.out_sr:
            speech = librosa.resample(speech, speech_sr, self.out_sr)
        speech = rms_normalize(speech)

        data, sr = sf.read(self.ir_directory_list[rir_idx % len(self.ir_directory_list)])
        if len(data.shape) == 2:
            data = data[..., 0]
        onset = get_onset(data)
        if onset > 10:
            randint = random.randrange(0, 10)
            data = np.concatenate([data[onset - randint:], np.zeros(onset - randint)])
        if self.out_sr is not None:
            if sr != self.out_sr:
                sig_len = len(data) / sr
                data = librosa.resample(data, sr, self.out_sr)
        data = data / np.sqrt(np.sum(data ** 2) + 1e-12)  ### NEW!!

        cspeech = signal.convolve(speech, data)
        return data, speech, cspeech



def ALTI_SPLIT(speech_lists, directory = '/SSD/RIR/altiverb-ir', split_ratio = (.6, .2), aug_const = 1, train_aug = False, out_len = 2, sr = 16000):
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

    train_alti = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = train_aug, out_len = out_len, out_sr = sr)
    vali_alti = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)
    test_alti = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)

    return train_alti, vali_alti, test_alti


class ALTIVERB(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = '/SSD/RIR/altiverb-ir',
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*/*/*.wav')

        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def ACE_SPLIT(speech_lists, directory = '/SSD/RIR/ACE', split_ratio = (.6, .2), train_aug = False, aug_const = 1, out_len = 2, sr = 16000):
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

    train_ace = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = train_aug, out_len = out_len, out_sr = sr)
    vali_ace = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)
    test_ace = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)

    return train_ace, vali_ace, test_ace


class ACE(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = '/SSD/RIR/ACE',
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*.wav')

        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def BUTREVDB_SPLIT(speech_lists, directory = '/SSD/BUTREVDB/RIR_only', split_ratio = (.6, .2), train_aug = False, out_len = 2, sr = 16000):
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

    train_but = IRDataset(train_directory_list, speech_lists[0], aug = train_aug, out_len = out_len, out_sr = sr)
    vali_but = IRDataset(vali_directory_list, speech_lists[1], aug = False, out_len = out_len, out_sr = sr)
    test_but = IRDataset(test_directory_list, speech_lists[2], aug = False, out_len = out_len, out_sr = sr)

    return train_but, vali_but, test_but

class BUTREVDB(IRDataset):
    def __init__(self,
                 directory = "/SSD/BUTREVDB/RIR_only",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*/*/*/*/RIR/*.wav')
        
        super().__init__(self.ir_directory_list, param_extract, out_sr, out_len)


class GREATHALL(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/RIR/greathall",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


class MARDY(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/RIR/MARDY",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2,
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
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


def OPENAIR_SPLIT(speech_lists, directory = '/SSD/RIR/OpenAIR', split_ratio = (.6, .2), train_aug = False, aug_const = 1, out_len = 2, sr = 16000):
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

    train_openair = IRDataset(train_directory_list, speech_lists[0], aug_const = aug_const, aug = train_aug, out_len = out_len, out_sr = sr)
    vali_openair = IRDataset(vali_directory_list, speech_lists[1], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)
    test_openair = IRDataset(test_directory_list, speech_lists[2], aug_const = 1, aug = False, out_len = out_len, out_sr = sr)

    return train_openair, vali_openair, test_openair


class OPENAIR(IRDataset):
    def __init__(self,
                 directory = "/SSD/RIR/OpenAIR",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2):

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
                 out_len = 2):

        self.ir_directory_list = glob.glob(directory + '/*/*.wav')
        
        super().__init__(self.ir_directory_list, speech_directory_list, param_extract, out_sr, out_len)


class RC24(IRDataset):
    def __init__(self,
                 speech_directory_list,
                 directory = "/SSD/ARIR/rc24",
                 param_extract = False,
                 out_sr = 16000,
                 out_len = 2,
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
                 out_len = 2,
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
                 out_len = 2,
                 sample_ratio = .01):

        self.ir_directory_list = glob.glob(directory + 'simulated_rirs/*/*/*.wav')
        
        super().__init__(self.ir_directory_list, 
                         speech_directory_list, 
                         param_extract, 
                         out_sr, 
                         out_len, 
                         sample_ratio = sample_ratio)


