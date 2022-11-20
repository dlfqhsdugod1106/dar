import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
import numpy as np 
import scipy.signal as signal
from sap import SAP
from svf import SVF
import util
import dsp

class FVN(nn.Module):
    def __init__(self, 
                 seg_num=160,
                 gain_per_seg=10,
                 ir_length=16000,
                 nonuniform_segmentation=False,
                 pulse_distance=10,
                 nonuniform_pulse_distance=False,
                 num_sap=20,
                 cumulative_sap=False,
                 device='cuda'):
        super().__init__()
        
        self.seg_num = seg_num
        self.gain_per_seg = gain_per_seg
        self.ir_length = ir_length
        self.nonuniform_segmentation = nonuniform_segmentation
        self.pulse_distance = pulse_distance
        self.nonuniform_pulse_distance = nonuniform_pulse_distance
        self.num_sap = num_sap
        self.cumulative_sap = cumulative_sap
        self.device = device
        self.pulse_distances = [10, 20, 35, 50, 65, 90, 120, 135, 180, 220, 
                                270, 320, 370, 420, 480, 540, 610, 680, 750, 820]

        if self.num_sap != 0:
            self.sap = self.schroeder_allpass(cumulative = self.cumulative_sap)

    def forward(self, z, color):
        gains, fir = z['g'], z['fir']
        batch_num = gains.shape[0]

        velvets = self.get_velvet_segments(batch_num, 
                                           self.ir_length, 
                                           self.seg_num, 
                                           self.gain_per_seg, 
                                           self.nonuniform_segmentation, 
                                           self.nonuniform_pulse_distance,
                                           self.pulse_distance)

        velvets = self.append_gain_and_merge(velvets, 
                                             gains,
                                             batch_num,
                                             self.seg_num,
                                             self.nonuniform_segmentation)

        if color != None:
            color = torch.fft.irfft(color)
            if self.num_sap != 0:
                if self.cumulative_sap:
                    color = dsp.convolve(color, self.sap.view(1, self.seg_num, -1))
                else:
                    color = dsp.convolve(color, self.sap.view(1, 1, -1))
        else:
            if self.cumulative_sap:
                color = self.sap.view(1, self.seg_num, -1)
            else:
                color = self.sap.view(1, 1, -1)

        colored = self.convolve(velvets, color, self.seg_num, self.nonuniform_segmentation)
        fvn = self.overlap_add(colored, self.seg_num, self.ir_length, self.nonuniform_segmentation)
        fir = F.pad(fir, (0, fvn.shape[-1]-fir.shape[-1]))
        fvn = fvn+fir
        return fvn

    def convolve(self, 
                 segments,
                 colors,
                 num_segments,
                 nonuniform_segmentation):

        if nonuniform_segmentation:
            segments[0] = dsp.convolve(segments[0], colors[:, :num_segments // 2, :])
            segments[1] = dsp.convolve(segments[1], colors[:, num_segments // 2 : 3 * num_segments // 4, :])
            segments[2] = dsp.convolve(segments[2], colors[:, 3 * num_segments // 4 :, :])
        else:
            segments = dsp.convolve(segments, colors)
        return segments

    def append_gain_and_merge(self,
                              segments,
                              gains,
                              batch_num,
                              num_segments,
                              nonuniform_segmentation):

        if gains.ndim == 2: 
            gains = gains.view(batch_num, num_segments, 1, 1)
        else:
            gains = gains.view(batch_num, num_segments, -1, 1)

        if nonuniform_segmentation:
            segments[0] = (segments[0] * gains[:, :num_segments // 2, :, :]).view(batch_num, num_segments // 2, -1)
            segments[1] = (segments[1] * gains[:, num_segments // 2: 3 * num_segments // 4, :, :]).view(batch_num, num_segments // 4, -1)
            segments[2] = (segments[2] * gains[:, 3 * num_segments // 4:, :, :]).view(batch_num, num_segments // 4, -1)

        else:
            segments = (segments * gains).view(batch_num, num_segments, -1)

        return segments


    def overlap_add(self, 
                    segments,
                    num_segments,
                    ir_length,
                    nonuniform_segmentation):

        if nonuniform_segmentation:
            stride = ir_length // num_segments // 2
            small_olaed = [dsp.overlap_add(segments[0].float(), stride),
                           dsp.overlap_add(segments[1].float(), stride * 2),
                           dsp.overlap_add(segments[2].float(), stride * 4)]
            small_olaed = [F.pad(small_olaed[2], (ir_length // 2, 0)),
                           F.pad(small_olaed[1], (ir_length // 4, small_olaed[2].shape[-1] - small_olaed[1].shape[-1] + ir_length // 4)),
                           F.pad(small_olaed[0], (0, small_olaed[2].shape[-1] - small_olaed[0].shape[-1] + ir_length // 2))]
            olaed = sum(small_olaed)

        else:
            stride = ir_length // num_segments
            olaed = dsp.overlap_add(segments, stride)

        return olaed

    def get_velvet_segments(self, 
                            batch_num, 
                            noise_length, 
                            num_segments, 
                            gain_per_segment,
                            nonuniform_segmentation, 
                            nonuniform_pulse_distance, 
                            pulse_distance):

        if nonuniform_segmentation:
            if nonuniform_pulse_distance:
                velvet_segments = [torch.stack([self.velvet_noise(batch_num, noise_length // 4 // (num_segments // 2), pulse_distance)\
                                   for pulse_distance in self.pulse_distances[:num_segments // 2]], -2)\
                                   .view(batch_num, num_segments // 2, gain_per_segment, -1)]\
                                  + [torch.stack([self.velvet_noise(batch_num, noise_length // 4 // (num_segments // 4), pulse_distance)\
                                   for pulse_distance in self.pulse_distances[num_segments // 2 : 3 * num_segments // 4]], -2)\
                                   .view(batch_num, num_segments // 4, gain_per_segment, -1)]\
                                  + [torch.stack([self.velvet_noise(batch_num, noise_length // 2 // (num_segments // 4), pulse_distance)\
                                   for pulse_distance in self.pulse_distances[3 * num_segments // 4:]], -2)\
                                   .view(batch_num, num_segments // 4, gain_per_segment, -1)]
            else:
                velvet_segments = [self.velvet_noise(batch_num, noise_length//4, pulse_distance).view(batch_num, num_segments//2, gain_per_segment, -1),
                                   self.velvet_noise(batch_num, noise_length//4, pulse_distance).view(batch_num, num_segments//4, gain_per_segment, -1),
                                   self.velvet_noise(batch_num, noise_length//2, pulse_distance).view(batch_num, num_segments//4, gain_per_segment, -1)]
        else:
            if nonuniform_pulse_distance:
                velvet_segments = torch.stack([self.velvet_noise(batch_num, noise_length // num_segments, pulse_distance) for pulse_distance in self.pulse_distances], -2)
                velvet_segments = velvet_segments.view(batch_num, num_segments, gain_per_segment, -1)
            else:
                velvet_segments = self.velvet_noise(batch_num, noise_length, pulse_distance).view(batch_num, num_segments, gain_per_segment, -1)

        return velvet_segments

    def velvet_noise(self, batch_num, noise_length, pulse_distance):
        chunks = (int(noise_length / pulse_distance) + 1) * batch_num
        
        position = torch.randint(0, pulse_distance, (chunks,), device = self.device)
        sparse_i = torch.stack([torch.arange(chunks).to(self.device), position], 0)
        sparse_v = torch.randint(0, 2, (chunks,), device = self.device) * 2 - 1
        
        velvet = torch.sparse.FloatTensor(sparse_i, sparse_v)
        velvet = velvet.to_dense()
        velvet = F.pad(velvet, (0, pulse_distance - velvet.shape[-1]))
        velvet = velvet.view(batch_num, -1)[:, :noise_length].float()
        return velvet
    
    def schroeder_allpass(self, 
                          As = list(np.linspace(0.75, 0.95, 20)),
                          Ns = [23, 48, 79, 109, 113, 
                                127, 163, 191, 229, 251, 
                                293, 337, 397, 421, 449,
                                509, 541, 601, 641, 691],
                          cumulative = False):

        Ns = Ns[:self.num_sap]
        allpasses = []
        allpass = np.zeros(9000)
        allpass[0] = 1.
        allpasses.append(allpass)

        for i in range(len(Ns)):
            B = np.zeros(Ns[i] + 1)
            B[0] = As[i]
            B[-1] = 1

            A = np.zeros(Ns[i] + 1)
            A[0] = 1
            A[-1] = As[i]
            
            allpasses.append(signal.lfilter(B, A, allpasses[-1]))

        if cumulative:
            allpass = torch.tensor(np.stack(allpasses[1:], 0)).float().to(self.device)
            allpass = torch.cat([allpass, allpass[-1:].repeat(self.seg_num - self.num_sap, 1)], 0)
        else:
            allpass = torch.tensor(allpasses[-1]).float().to(self.device)
        return allpass

class FVN_Decoder(nn.Module):
    def __init__(self,
            input_shape = (500, 64),
            num_seg = 160,
            gain_per_seg = 10,
            fir_len = 100,
            g_db = False):

        super().__init__()

        self.g_decoder = util.LinearLinear(input_shape, (num_seg, gain_per_seg))
        self.fir_decoder = util.LinearLinear(input_shape, (fir_len, 1))
        self.g_db = g_db

    def forward(self, latent):
        g = self.g_decoder(latent).squeeze(-1)
        if self.g_db:
            g = 10 ** (- F.softplus(g) / 20)
        fir = self.fir_decoder(latent).squeeze(-1)
        params = dict(g = g, fir = fir)
        return params
