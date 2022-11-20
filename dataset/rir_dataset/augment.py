import numpy as np
import scipy.signal as signal
import random

def augment(ir,
            da_strength_range = (-6, 6),  # in dB
            fb_band = 'bark',
            geq_gain_range = (-3, 3), # in dB
            geq_smoothing = 3,#2,
            lr_tau_range = (0, 6),#(-2, 4),
            lr_fb_tau_range = (0, 3),
            lr_fb_tau_smoothing = 3):

    onset = get_onset(ir)

    #if onset > 100:
    #    randint = random.randrange(0, 100)
    #    ir = np.concatenate([ir[onset - randint:], np.zeros(onset - randint)])
    #    onset = randint

    da_window, lr_window = splitter_window(onset)

    da_ir, lr_ir = ir * da_window, ir * lr_window
    da_fbir, lr_fbir = fbir(da_ir, band = 'bark'), fbir(lr_ir, band = 'bark')
    num_fb = lr_fbir.shape[0]

    da_strength_db = np.random.rand(1) * (da_strength_range[1] - da_strength_range[0]) + da_strength_range[0] # uniform dist. 
    da_strength = 10 ** (da_strength_db / 20)
    
    geq_gain_db = np.random.rand(num_fb + geq_smoothing - 1) * (geq_gain_range[1] - geq_gain_range[0]) + geq_gain_range[0] # uniform dist.
    geq_gain_smoother = np.hanning(geq_smoothing + 2)
    geq_gain_smoother = geq_gain_smoother / np.sum(geq_gain_smoother)
    geq_gain_db = np.convolve(geq_gain_db, geq_gain_smoother)[geq_smoothing:-geq_smoothing] 
    geq_gain = 10 ** (geq_gain_db / 20)

    lr_tau = np.random.rand(1) * (lr_tau_range[1] - lr_tau_range[0]) + lr_tau_range[0]
    lr_env = decay_curve(onset, lr_tau)

    lr_fb_taus = np.random.rand(num_fb + lr_fb_tau_smoothing - 1) * (lr_fb_tau_range[1] - lr_fb_tau_range[0]) + lr_fb_tau_range[0]
    lr_fb_taus_smoother = np.hanning(lr_fb_tau_smoothing + 2)
    lr_fb_taus_smoother = lr_fb_taus_smoother / np.sum(lr_fb_taus_smoother)
    lr_fb_taus = np.convolve(lr_fb_taus, lr_fb_taus_smoother)[lr_fb_tau_smoothing:-lr_fb_tau_smoothing] 
    lr_fb_env = decay_curve(onset, lr_fb_taus)

    #aug_da = np.sum(da_fbir * geq_gain[:, np.newaxis], axis = 0) * da_strength !!!
    aug_da = np.sum(da_fbir, axis = 0) * da_strength
    aug_lr = np.sum(lr_fbir * geq_gain[:, np.newaxis] * lr_fb_env, axis = 0) * lr_env

    aug_ir = aug_da + aug_lr
    aug_ir = aug_ir.squeeze()
    #aug_ir = aug_ir / np.sum(aug_ir ** 2 + 1e-6)

    return aug_ir


def speech_augment(speech,
                   fb_band = 'bark',
                   geq_gain_range = (-3, 3)):

    speech_fb = fbir(speech, band = 'bark')
    
    num_fb = speech_fb.shape[0]

    geq_smoothing = np.random.randint(1, 4)
    geq_gain_db = np.random.uniform(geq_gain_range[0], geq_gain_range[1], size = (num_fb + geq_smoothing - 1,))
    geq_gain_smoother = np.hanning(geq_smoothing + 2)
    geq_gain_smoother = geq_gain_smoother / np.sum(geq_gain_smoother)
    geq_gain_db = np.convolve(geq_gain_db, geq_gain_smoother)[geq_smoothing:-geq_smoothing]
    geq_gain = 10 ** (geq_gain_db / 20)

    speech_aug = np.sum(speech_fb * geq_gain[:, np.newaxis], axis = 0)

    return speech_aug


def augment2(ir,
            da_strength_range = (-6, 6),  # in dB
            fb_band = 'bark',
            geq_gain_range = (-6, 6), # in dB
            geq_smoothing = 2,
            lr_tau_range = (-2, 8),#(-2, 4),
            lr_fb_tau_range = (-1, 4),
            lr_fb_tau_smoothing = 2):

    onset = get_onset(ir)

    #if onset > 100:
    #    randint = random.randrange(0, 100)
    #    ir = np.concatenate([ir[onset - randint:], np.zeros(onset - randint)])
    #    onset = randint

    da_window, lr_window = splitter_window(onset)

    da_ir, lr_ir = ir * da_window, ir * lr_window
    da_fbir, lr_fbir = fbir(da_ir, band = 'bark'), fbir(lr_ir, band = 'bark')
    num_fb = lr_fbir.shape[0]

    da_strength_db = np.random.rand(1) * (da_strength_range[1] - da_strength_range[0]) + da_strength_range[0] # uniform dist. 
    da_strength = 10 ** (da_strength_db / 20)
    
    geq_gain_db = np.random.rand(num_fb + geq_smoothing - 1) * (geq_gain_range[1] - geq_gain_range[0]) + geq_gain_range[0] # uniform dist.
    geq_gain_smoother = np.hanning(geq_smoothing + 2)
    geq_gain_smoother = geq_gain_smoother / np.sum(geq_gain_smoother)
    geq_gain_db = np.convolve(geq_gain_db, geq_gain_smoother)[geq_smoothing:-geq_smoothing] 
    geq_gain = 10 ** (geq_gain_db / 20)

    lr_tau = np.random.rand(1) * (lr_tau_range[1] - lr_tau_range[0]) + lr_tau_range[0]
    lr_env = decay_curve(onset, lr_tau)

    lr_fb_taus = np.random.rand(num_fb + lr_fb_tau_smoothing - 1) * (lr_fb_tau_range[1] - lr_fb_tau_range[0]) + lr_fb_tau_range[0]
    lr_fb_taus_smoother = np.hanning(lr_fb_tau_smoothing + 2)
    lr_fb_taus_smoother = lr_fb_taus_smoother / np.sum(lr_fb_taus_smoother)
    lr_fb_taus = np.convolve(lr_fb_taus, lr_fb_taus_smoother)[lr_fb_tau_smoothing:-lr_fb_tau_smoothing] 
    lr_fb_env = decay_curve(onset, lr_fb_taus)

    aug_da = np.sum(da_fbir * geq_gain[:, np.newaxis], axis = 0) * da_strength
    aug_lr = np.sum(lr_fbir * geq_gain[:, np.newaxis] * lr_fb_env, axis = 0) * lr_env

    aug_ir = aug_da + aug_lr
    aug_ir = aug_ir.squeeze()
    #aug_ir = aug_ir / np.sum(aug_ir ** 2 + 1e-6)

    return aug_ir

def fbir(ir, band = 'oct', ir_ms = 1000):
    sr = ir.shape[-1] * 1e3 / ir_ms

    if band == 'bark':
        low, hi = bark_band(sr)
    elif band == 'oct':
        low, hi = oct_band(sr)

    f_ir = []

    for low_f, hi_f in zip(low, hi):
        sos = signal.butter(N = 4,
                Wn = np.array([low_f, hi_f]) / sr * 2,
                btype = 'bandpass',
                analog = False,
                output = 'sos')

        f_ir.append(signal.sosfiltfilt(sos, ir).copy())

    f_ir = np.stack(f_ir, axis = 0)

    return f_ir


def get_onset(ir):
    power = ir ** 2
    window = np.hanning(128)
    window /= np.sum(window)
    smoothed = np.convolve(power, window)[:len(ir)]
    max_pow_idx = np.argmax(smoothed)
    atk = smoothed[:np.argmax(smoothed) + 2]
    ratio = atk[1:] / (atk[:-1] + 1e-5)
    onset = np.argmax(ratio)
    return onset


def splitter_window(onset, num_sample = 16000, sr = 16000):

    window_sample = int(sr * 5 * 1e-3)
    da_window = np.hanning(window_sample)

    if onset < window_sample // 2:
        da_window = da_window[window_sample // 2 - onset:]
        da_window = np.concatenate([da_window, np.zeros(num_sample - len(da_window))])
    elif num_sample - onset - window_sample // 2 >= 0:
        da_window = np.concatenate([np.zeros(onset - window_sample // 2), 
                                    da_window, 
                                    np.zeros(num_sample - onset - window_sample // 2)])
    else:
        da_window = np.zeros(num_sample)

    lr_window = np.ones(num_sample) - da_window

    return da_window, lr_window


def decay_curve(onset, tau, num_sample = 16000):
    decay = np.linspace(0, 1, num_sample - onset)
    if type(tau) == np.ndarray:
        env = np.exp(- decay[np.newaxis, :] * tau[:, np.newaxis])
        env = np.concatenate([np.ones((env.shape[0], onset)), env], axis = 1)
    else:
        env = np.exp(- decay * tau)
        env = np.concatenate([np.ones(onset), env])

    return env


def bark_band(sr):
    """
    Returns Bark Band Cutoff Frequencies
    """

    freq = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 
                     1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 
                     3700, 4400, 5300, 6400, 7700, 9500, 12000, 16000]) 

    # last freq is modified (orig. 15500)

    freq = freq[freq < sr / 2]
    if sr / 2 - freq[-1] < (freq[-1] - freq[-2]) / 2:
        freq = freq[:-1]

    low, hi = freq[:-1], freq[1:]

    return low, hi


def oct_band(sr):
    """
    Returns Octave Band Cutoff Frequencies
    """

    freq = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])

    freq = freq[freq < sr / 2]
    if sr / 2 - freq[-1] < (freq[-1] - freq[-2]) / 2:
        freq = freq[:-1]

    low, hi = freq / np.sqrt(2), freq * np.sqrt(2)

    return low, hi
