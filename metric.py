import numpy as np
from scipy import signal
from scipy import stats
from nnAudio import features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio

def oct_band(sr):
    freq = np.array([125, 250, 500, 1000, 2000, 4000, 8000])

    freq = freq[freq < sr / 2]
    if sr / 2 - freq[-1] < (freq[-1] - freq[-2]) / 2:
        freq = freq[:-1]

    low, hi = freq / np.sqrt(2), freq * np.sqrt(2)

    return low, hi

def EDR(ir, sr = 16000, ir_ms = 2000, window_len = 128, band = 'oct'):
    num_batch = ir.shape[0]

    if band != 'none':
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

            f_ir.append(signal.sosfiltfilt(sos, ir))

        f_ir = np.stack(f_ir, 1)
    else:
        f_ir = np.stack([ir], 1)

    power = f_ir ** 2

    #window = np.hanning(window_len + 1)[np.newaxis, np.newaxis, :]
    #window = window / np.sum(window)

    #smoothed = signal.convolve(power, window)[:, :, window_len // 2 : - window_len // 2]
    edr = np.flip(np.cumsum(np.flip(power, 2), 2), 2)

    taxis = np.arange(ir.shape[-1]) / ir.shape[-1] * ir_ms
    tedr = np.flip(np.cumsum(np.flip(power * taxis, 2), 2), 2)

    return edr, tedr

def cross_idx(logedr, db):
    num_batch = logedr.shape[0]
    logedr = logedr.reshape(-1, logedr.shape[-1])
    idxs = np.zeros(logedr.shape[0])
    for i in range(logedr.shape[0]):
        cross = np.where(logedr[i] < db)[0]
        if len(cross) == 0:
            idxs[i] = logedr.shape[-1] - 1
        else:
            idxs[i] = cross[0]
    idxs = idxs.reshape(num_batch, -1)
    return idxs

def RT(edr, ir_ms = 2000, eps = 1e-10):
    num_batch, num_bank = edr.shape[0], edr.shape[1]

    logedr = 10 * np.log10(edr + eps)
    logedr = logedr - logedr[:, :, 0][:, :, np.newaxis] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ct_5 = cross_idx(logedr, -5) / logedr.shape[-1] * ir_ms
    ct_10 = cross_idx(logedr, -10) / logedr.shape[-1] * ir_ms
    ct_15 = cross_idx(logedr, -15) / logedr.shape[-1] * ir_ms
    ct_25 = cross_idx(logedr, -25) / logedr.shape[-1] * ir_ms
    ct_35 = cross_idx(logedr, -35) / logedr.shape[-1] * ir_ms

    ct_5 = ct_5.reshape(-1)
    ct_10 = ct_10.reshape(-1)
    ct_15 = ct_15.reshape(-1)
    ct_25 = ct_25.reshape(-1)
    ct_35 = ct_35.reshape(-1)
    
    ct_10[ct_10 > ir_ms * .99] = ct_5[ct_10 > ir_ms * .99] * 2
    EDT = ct_10

    ct_15[ct_15 > ir_ms * .99] = ct_5[ct_15 > ir_ms * .99] * 3
    T10 = ct_15 - ct_5

    ct_25[ct_25 > ir_ms * .99] = ct_5[ct_25 > ir_ms * .99] + T10[ct_25 > ir_ms * .99] * 2
    T20 = ct_25 - ct_5
    #T20[T20 == 0] = ir_ms

    ct_35[ct_35 > ir_ms * .99] = ct_5[ct_35 > ir_ms * .99] + T20[ct_35 > ir_ms * .99] * 1.5
    T30 = ct_35 - ct_5
    #T30[T30 == 0] = ir_ms

    EDT = EDT.reshape(num_batch, num_bank)
    T20 = T20.reshape(num_batch, num_bank)
    T30 = T30.reshape(num_batch, num_bank)


    return dict(EDT = EDT, T20 = T20, T30 = T30)

def RT2(edr, ir_ms = 2000, sr = 16000, eps = 1e-10):
    num_batch, num_bank = edr.shape[0], edr.shape[1]

    logedr = 10 * np.log10(edr + eps)
    logedr = logedr - logedr[:, :, 0][:, :, np.newaxis] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ct_5 = cross_idx(logedr, -5).astype(int)
    ct_10 = cross_idx(logedr, -10).astype(int)
    ct_25 = cross_idx(logedr, -25).astype(int)
    ct_35 = cross_idx(logedr, -35).astype(int)

    EDT = np.zeros_like(ct_5).astype(float)
    T20 = np.zeros_like(ct_5).astype(float)
    T30 = np.zeros_like(ct_5).astype(float)

    for i in range(ct_5.shape[0]):
        for j in range(ct_5.shape[1]):
            edt_reg_range = logedr[i, j, :ct_10[i, j]]
            t20_reg_range = logedr[i, j, ct_5[i, j]:ct_25[i, j]]
            t30_reg_range = logedr[i, j, ct_5[i, j]:ct_35[i, j]]

            if edt_reg_range.shape[0] != 0:
                edt_p = stats.linregress(np.arange(edt_reg_range.shape[0]), edt_reg_range)
                EDT[i, j] = - 10 / edt_p.slope / sr * 1e3
            else:
                EDT[i, j] = 0

            if t20_reg_range.shape[0] != 0:
                t20_p = stats.linregress(np.arange(t20_reg_range.shape[0]), t20_reg_range)
                T20[i, j] = - 20 / t20_p.slope / sr * 1e3
            else:
                T30[i, j] = 0

            if t30_reg_range.shape[0] != 0:
                t30_p = stats.linregress(np.arange(t30_reg_range.shape[0]), t30_reg_range)
                T30[i, j] = - 30 / t30_p.slope / sr * 1e3
            else:
                T30[i, j] = 0

    return dict(EDT = EDT, T20 = T20, T30 = T30)

def DIFF_RT(rir_rt, recon_rt, eps = 1e-10, individual_out = False):
    diff, diff_avg, diff_worst = {}, {}, {}
    for key in rir_rt.keys():
        diff_ratio = np.abs(rir_rt[key] - recon_rt[key]) / (rir_rt[key] + eps) * 100

        diff[key] = diff_ratio
        diff_avg[key] = np.average(diff_ratio, 1)
        diff_worst[key] = np.max(diff_ratio, 1)

    if individual_out:
        return diff_avg, diff_worst, diff
    else:
        return diff_avg, diff_worst


def JND_RT(rir_rt, recon_rt, eps = 1e-10, individual_out = False):
    jndr, jndr_avg, jndr_worst = {}, {}, {}
    for key in rir_rt.keys():
        jnd_ratio = np.abs(rir_rt[key] - recon_rt[key]) / (rir_rt[key] * 0.05 + eps)

        jndr[key] = jnd_ratio
        jndr_avg[key] = np.average(jnd_ratio, 1)
        jndr_worst[key] = np.max(jnd_ratio, 1)

    if individual_out:
        return jndr_avg, jndr_worst, jndr
    else:
        return jndr_avg, jndr_worst

def eint(edr, t0, t, ir_ms = 2000):
    frame_ms = ir_ms / edr.shape[-1]

    t_frame_idx = int(t / frame_ms)
    if t == ir_ms: t_frame_idx -= 1
    t0_frame_idx = int(t0 / frame_ms)
    eint = edr[..., t0_frame_idx] - edr[..., t_frame_idx]

    return eint

def DEL(edr, tedr, ir_ms = 2000, eps = 1e-10):
    return dict(DRR       = 10 * np.log10(eint(edr, 0, 5, ir_ms) / (eint(edr, 5, ir_ms, ir_ms) + eps) + eps),
                D_50      = eint(edr, 0, 50, ir_ms) / (eint(edr, 0, ir_ms, ir_ms) + eps),
                D_80      = eint(edr, 0, 80, ir_ms) / (eint(edr, 0, ir_ms, ir_ms) + eps),
                C_50      = 10 * np.log10(eint(edr, 0, 50, ir_ms) / (eint(edr, 50, ir_ms, ir_ms) + eps) + eps),
                C_80      = 10 * np.log10(eint(edr, 0, 80, ir_ms) / (eint(edr, 80, ir_ms, ir_ms) + eps) + eps),
                T_S       = eint(tedr, 0, ir_ms, ir_ms) / (eint(edr, 0, ir_ms, ir_ms) + eps),
                ST_early  = 10 * np.log10(eint(edr, 20, 100, ir_ms) / (eint(edr, 0, 10, ir_ms) + eps) + eps),
                ST_late   = 10 * np.log10(eint(edr, 100, ir_ms, ir_ms) / (eint(edr, 0, 10, ir_ms) + eps) + eps))

def DRR_JND(DRR):
    """
    range   : DRR           : JND
    R0      : ~ -10db       : 6db
    R1      : -10db ~ 0db   : 6db to 2db
    R2      : 0db ~ 10db    : 2db to 3db
    R3      : 10 ~ 20db     : 3db to 8db
    R4      : 20db ~        : 8db
    """

    DRR_JND = np.zeros_like(DRR)

    DRR_JND[DRR < -10] = 6
    DRR_JND[(DRR >= -10) * (DRR < 0)] = 6 - 4 * (DRR[(DRR >= -10) * (DRR < 0)] + 10) / 10
    DRR_JND[(DRR >= 0) * (DRR < 10)] = 3 + (DRR[(DRR >= 0) * (DRR < 10)] - 10) / 10
    DRR_JND[(DRR >= 10) * (DRR < 20)] = 8 + 5 * (DRR[(DRR >= 10) * (DRR < 20)] - 20) / 10
    DRR_JND[DRR >= 20] = 8

    return DRR_JND

def DIFF_DEL(rir_del, recon_del, individual_out = False):
    diff_avg, diff_worst = {}, {}

    del_diffs = dict(DRR       = np.abs(recon_del['DRR'] - rir_del['DRR']),
                     #D_50      = np.abs(recon_del['D_50'] - rir_del['D_50']) / .05,
                     #D_80      = np.abs(recon_del['D_80'] - rir_del['D_80']) / .05,
                     C_50      = np.abs(recon_del['C_50'] - rir_del['C_50']),)
                     #C_80      = np.abs(recon_del['C_80'] - rir_del['C_80']),
                     #T_S       = np.abs(recon_del['T_S'] - rir_del['T_S']) / 10,
                     #ST_early  = np.abs(recon_del['ST_early'] - rir_del['ST_early']) / 2,
                     #ST_late   = np.abs(recon_del['ST_late'] - rir_del['ST_late']) / 2)

    for key in del_diffs.keys():
        diff_avg[key] = np.average(del_diffs[key], 1)
        diff_worst[key] = np.max(del_diffs[key], 1)

    if individual_out:
        return diff_avg, diff_worst, del_diffs
    else:
        return diff_avg, diff_worst


def JND_DEL(rir_del, recon_del, individual_out = False):
    jndr_avg, jndr_worst = {}, {}

    jnd_ratios = dict(DRR       = np.abs(recon_del['DRR'] - rir_del['DRR']) / DRR_JND(rir_del['DRR']),
                      D_50      = np.abs(recon_del['D_50'] - rir_del['D_50']) / .05,
                      D_80      = np.abs(recon_del['D_80'] - rir_del['D_80']) / .05,
                      C_50      = np.abs(recon_del['C_50'] - rir_del['C_50']),
                      C_80      = np.abs(recon_del['C_80'] - rir_del['C_80']),
                      T_S       = np.abs(recon_del['T_S'] - rir_del['T_S']) / 10,
                      ST_early  = np.abs(recon_del['ST_early'] - rir_del['ST_early']) / 2,
                      ST_late   = np.abs(recon_del['ST_late'] - rir_del['ST_late']) / 2)

    for key in jnd_ratios.keys():
        jndr_avg[key] = np.average(jnd_ratios[key], 1)
        jndr_worst[key] = np.max(jnd_ratios[key], 1)

    if individual_out:
        return jndr_avg, jndr_worst, jnd_ratios
    else:
        return jndr_avg, jndr_worst

def jndr_profile(rir, recon, sr = 16000):
    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, ir_ms = ir_ms)
    rir_rt = RT(rir_edr, ir_ms = ir_ms)
    rir_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    recon_edr, recon_tedr = EDR(recon, ir_ms = ir_ms)
    recon_rt = RT(recon_edr, ir_ms = ir_ms)
    recon_del = DEL(recon_edr, recon_tedr, ir_ms = ir_ms)

    jndr_rt_avg, jndr_rt_worst = JND_RT(rir_rt, recon_rt)
    jndr_del_avg, jndr_del_worst = JND_DEL(rir_del, recon_del)

    jndr_avg = {**jndr_rt_avg, **jndr_del_avg}
    jndr_worst = {**jndr_rt_worst, **jndr_del_worst}

    p_rir = {**rir_rt, **rir_del}
    p_recon = {**recon_rt, **recon_del}

    return jndr_avg, jndr_worst, p_rir, p_recon

def jndr_profile2(rir, recon, sr = 16000, individual_out = False):
    if sr == 48000:
        window_len = 512
    else:
        window_len = 128

    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, sr = sr, ir_ms = ir_ms, window_len = window_len)
    rir_rt = RT2(rir_edr, ir_ms = ir_ms, sr = sr)
    rir_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    recon_edr, recon_tedr = EDR(recon, sr = sr, ir_ms = ir_ms, window_len = window_len)
    recon_rt = RT2(recon_edr, ir_ms = ir_ms, sr = sr)
    recon_del = DEL(recon_edr, recon_tedr, ir_ms = ir_ms)

    if individual_out:
        jndr_rt_avg, jndr_rt_worst, jndr_rt = JND_RT(rir_rt, recon_rt, individual_out = True)
        jndr_del_avg, jndr_del_worst, jndr_del = JND_DEL(rir_del, recon_del, individual_out = True)
        jndr = {**jndr_rt, **jndr_del}
    else:
        jndr_rt_avg, jndr_rt_worst = JND_RT(rir_rt, recon_rt)
        jndr_del_avg, jndr_del_worst = JND_DEL(rir_del, recon_del)

    jndr_avg = {**jndr_rt_avg, **jndr_del_avg}
    jndr_worst = {**jndr_rt_worst, **jndr_del_worst}

    p_rir = {**rir_rt, **rir_del}
    p_recon = {**recon_rt, **recon_del}

    if individual_out:
        return jndr_avg, jndr_worst, jndr, p_rir, p_recon
    else:
        return jndr_avg, jndr_worst, p_rir, p_recon

def jndr_profile3(rir, recon, sr = 16000):
    if sr == 48000:
        window_len = 512
    else:
        window_len = 128

    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, sr = sr, ir_ms = ir_ms, window_len = window_len, band = 'none')
    rir_rt = RT2(rir_edr, ir_ms = ir_ms, sr = sr)
    rir_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    recon_edr, recon_tedr = EDR(recon, sr = sr, ir_ms = ir_ms, window_len = window_len, band = 'none')
    recon_rt = RT2(recon_edr, ir_ms = ir_ms, sr = sr)
    recon_del = DEL(recon_edr, recon_tedr, ir_ms = ir_ms)

    jndr_rt_avg, jndr_rt_worst = JND_RT(rir_rt, recon_rt)
    jndr_del_avg, jndr_del_worst = JND_DEL(rir_del, recon_del)

    jndr_avg = {**jndr_rt_avg, **jndr_del_avg}
    jndr_worst = {**jndr_rt_worst, **jndr_del_worst}

    p_rir = {**rir_rt, **rir_del}
    p_recon = {**recon_rt, **recon_del}

    return jndr_avg, jndr_worst, p_rir, p_recon



class LogEDRDifference(nn.Module):
    def __init__(self,):
        super().__init__() 
        self.logspec = features.STFT(sr=48000, n_fft=2048, hop_length=2048, freq_scale='log', fmin=60, fmax=48000/2, output_format='Magnitude')

    def forward(self, rir, recon):
        rir_edr = self.get_edr_e(rir)
        recon_edr = self.get_edr_e(recon)
        edr_error = torch.mean(torch.abs(10*torch.log10((rir_edr+1e-5)/(recon_edr+1e-5))))
        return edr_error

    def get_edr_e(self, ir):
        D = logspec(ir)**2
        D = np.cumsum(D[..., ::-1], -1)[..., ::-1]
        return D 


def difference_profile(rir, recon, sr = 16000):
    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, sr = sr, ir_ms = ir_ms)
    rir_rt = RT2(rir_edr, ir_ms = ir_ms, sr = sr)
    rir_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    recon_edr, recon_tedr = EDR(recon, sr = sr, ir_ms = ir_ms)
    recon_rt = RT2(recon_edr, ir_ms = ir_ms, sr = sr)
    recon_del = DEL(recon_edr, recon_tedr, ir_ms = ir_ms)

    rir_p = {**rir_rt, **rir_del}
    recon_p = {**recon_rt, **recon_del}

    freq_rt_avg, freq_rt_worst = DIFF_RT(rir_rt, recon_rt)
    freq_del_avg, freq_del_worst = DIFF_DEL(rir_del, recon_del)

    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, sr = sr, ir_ms = ir_ms, band = 'none')
    rir_full_rt = RT2(rir_edr, ir_ms = ir_ms, sr = sr)
    rir_full_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    recon_edr, recon_tedr = EDR(recon, sr = sr, ir_ms = ir_ms, band = 'none')
    recon_full_rt = RT2(recon_edr, ir_ms = ir_ms, sr = sr)
    recon_full_del = DEL(recon_edr, recon_tedr, ir_ms = ir_ms)

    full_rt_avg, _ = DIFF_RT(rir_full_rt, recon_full_rt)
    full_del_avg, _ = DIFF_DEL(rir_full_del, recon_full_del)

    rir_full_p = {**rir_full_rt, **rir_full_del}
    recon_full_p = {**recon_full_rt, **recon_full_del}

    diff_full = {**full_rt_avg, **full_del_avg}
    diff_freq_avg = {**freq_rt_avg, **freq_del_avg}
    diff_freq_worst = {**freq_rt_worst, **freq_del_worst}

    return diff_full, diff_freq_avg, diff_freq_worst, rir_full_p, recon_full_p, rir_p, recon_p

def params_profile(rir, sr = 16000):
    ir_ms = rir.shape[-1] / sr * 1e3
    rir_edr, rir_tedr = EDR(rir, sr = sr, ir_ms = ir_ms)
    rir_rt = RT2(rir_edr, ir_ms = ir_ms, sr = sr)
    rir_del = DEL(rir_edr, rir_tedr, ir_ms = ir_ms)

    params = {**rir_rt, **rir_del}
    return params 
