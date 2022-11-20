import numpy as np
import soundfile as sf
from scipy import signal
import torch
from torch.utils.data import Dataset, ConcatDataset
from glob import glob
import random
import librosa
import pickle

def generate_dataset(ir_len = 2, noiseir_ratio = .3, sr = 16000, speech_len = 2.5, ism_max_order = 50, vali_dspeech_idx = False, time_alias = False):
    train_speech, vali_speech, test_speech = get_speech_lists()

    vali_alti, test_alti_1, test_alti_2 = ALTI_SPLIT(split_ratio = (.5, .2))
    vali_ace, test_ace_1, test_ace_2 = ACE_SPLIT(split_ratio = (.5, .2))
    vali_open, test_open_1, test_open_2 = OPENAIR_SPLIT(split_ratio = (.5, .2))

    vali_list, test_list = pickle.load(open('rir_dir.pickle', 'rb'))

    train_ir = PyRoom3_saved(train_speech, sr=sr, ir_len=ir_len, noiseir_ratio=noiseir_ratio, speech_len=speech_len)
    vali_ir = IRDataset(vali_list, vali_speech, out_sr=sr, out_len=ir_len, speech_len=speech_len, get_speech_idx=vali_dspeech_idx)
    test_ir = IRDataset(test_list, test_speech, out_sr=sr, out_len=ir_len, speech_len=speech_len, get_speech_idx=True)
    return train_ir, vali_ir, test_ir

def get_vctk(dataset_dir='/SSD/VCTK/vctk', train_ratio=0.7, vali_ratio=0.25):
    spk_list = glob(dataset_dir + '/wav48_silence_trimmed/*')
    split_idx_1 = int(len(spk_list) * train_ratio)
    split_idx_2 = int(len(spk_list) * (train_ratio + vali_ratio))
    train_spk_div = spk_list[:split_idx_1]
    vali_spk_div = spk_list[split_idx_1:split_idx_2]
    test_spk_div = spk_list[split_idx_2:]

    train_list = []
    for spk in train_spk_div:
        train_list += glob(spk + '/*.flac')

    vali_list = []
    for spk in vali_spk_div:
        vali_list += glob(spk + '/*.flac')

    test_list = []
    for spk in test_spk_div:
        test_list += glob(spk + '/*.flac')

    print("VCTK :", len(train_list), len(vali_list), len(test_list))

    return train_list, vali_list, test_list

def get_speech_lists():
    train_vctk, vali_vctk, test_vctk = get_vctk()
    train_speech = train_vctk 
    vali_speech = vali_vctk 
    test_speech = test_vctk 
    random.shuffle(train_speech)
    random.shuffle(vali_speech)
    random.shuffle(test_speech)
    return train_speech, vali_speech, test_speech

class PyRoom3_saved(Dataset):
    def __init__(self, 
                 speech_directory_list,
                 pyroom_directory = '/HDD1/pyroom_data/',
                 sr = 16000,
                 ir_len = 2.5,
                 speech_len = 3,
                 noiseir_ratio = 0.3,
                 rdeq = True,
                 rdeq_range = (2, 6),
                 time_alias = False):

        super().__init__()

        self.speech_directory_list = speech_directory_list
        self.speech_dataset_len = len(self.speech_directory_list)

        self.ir_directory = sorted(glob(pyroom_directory + '*/*.npy'))

        self.sr = sr
        self.ir_len = ir_len 
        self.speech_len = speech_len
        self.noiseir_ratio = noiseir_ratio
        self.rdeq = rdeq
        self.rdeq_range = rdeq_range
        self.time_alias = time_alias


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
            t0 = random.randint(0, margin - 1)
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
                 time_alias = False,
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
        self.time_alias = time_alias
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

        if self.time_alias:
            crop_data = self.gt_time_alias(data, int(self.out_sr * self.out_len))
        else:
            crop_data = data[:int(self.out_sr * self.out_len)]

        dryspeech_cut = dryspeech_cut[len(crop_data):]
        tspeech = tspeech[len(crop_data):]

        if self.get_speech_idx:
            if self.time_alias:
                return crop_data, tspeech, dryspeech_cut, speech_idx, data[:int(self.out_sr * self.out_len)]
            else:
                return crop_data, tspeech, dryspeech_cut, speech_idx
        else:
            if self.time_alias:
                return crop_data, tspeech, dryspeech_cut, data[:int(self.out_sr * self.out_len)]
            else:
                return crop_data, tspeech, dryspeech_cut

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

    def gt_time_alias(self, x, L):
        x = np.pad(x, (0, (L - x.shape[-1] % L) % L))
        alias_x = np.split(x, x.shape[-1] // L)
        alias_x = sum(alias_x)
        return alias_x

def ALTI_SPLIT(directory = '/SSD/RIR/altiverb-ir', split_ratio = (.5, .2)):
    total_directory_list = glob(directory + '/*/*/*.wav')

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
        train_directory_list += glob(train_place + '*.wav')

    vali_directory_list = []
    for vali_place in vali_place_list:
        vali_directory_list += glob(vali_place + '*.wav')

    test_directory_list = []
    for test_place in test_place_list:
        test_directory_list += glob(test_place + '*.wav')

    return train_directory_list, vali_directory_list, test_directory_list

def ACE_SPLIT(directory = '/SSD/RIR/ACE', split_ratio = (.5, .2)):
    total_directory_list = glob(directory + '/*.wav')

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
        train_directory_list += glob(train_place + '*.wav')

    vali_directory_list = []
    for vali_place in vali_place_list:
        vali_directory_list += glob(vali_place + '*.wav')

    test_directory_list = []
    for test_place in test_place_list:
        test_directory_list += glob(test_place + '*.wav')

    return train_directory_list, vali_directory_list, test_directory_list

def OPENAIR_SPLIT(directory = '/SSD/RIR/OpenAIR', split_ratio = (.5, .2)):
    place_list = glob(directory + '/*')
    random.shuffle(place_list)

    num_train = int(len(place_list) * split_ratio[0])
    num_vali = int(len(place_list) * split_ratio[1])

    train_place_list = place_list[:num_train]
    vali_place_list = place_list[num_train:num_train + num_vali]
    test_place_list = place_list[num_train + num_vali:] 

    mic_channel = ['mono', 'stereo', 'b-format', 'surround-5-1']

    train_directory_list = []
    vali_directory_list = []
    test_directory_list = []

    for train_place in train_place_list:
        mic_list = [glob(train_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob(mic_dir[0] + '/*.wav')
                train_directory_list += ir_directory
                break

    for vali_place in vali_place_list:
        mic_list = [glob(vali_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob(mic_dir[0] + '/*.wav')
                vali_directory_list += ir_directory
                break

    for test_place in test_place_list:
        mic_list = [glob(test_place + '/' + mic) for mic in mic_channel]
        for mic_dir in mic_list:
            if len(mic_dir) != 0:
                ir_directory = glob(mic_dir[0] + '/*.wav')
                test_directory_list += ir_directory
                break

    return train_directory_list, vali_directory_list, test_directory_list
