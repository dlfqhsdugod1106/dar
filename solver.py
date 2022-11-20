import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset 
import torch.autograd.profiler as profiler

import numpy as np
import soundfile as sf

from components import core_complex as comp 
from components import core_dsp as dsp
from components import filter_complex as filt
from components import utils as util
from components.fvn_complex_2 import FVN2, TrueFVN
from components.svf import SVF
from components.biquad_complex import Biquad

from network.fvn_decoder import FVN_Decoder as FVN_Decoder
from network.svf_decoder import SVF_Decoder as SVF_Decoder
from network.svf_decoder import SVFParametricEQ_Decoder_Stable_Shelving3 as SVFParametricEQ_Decoder
from network.biq_decoder import Biquad_Decoder as Biquad_Decoder
from network.fir_decoder import FIR_Decoder, FIR_LinPhase_Decoder
from network.encoder_rnn_optimized_2 import Encoder
from loss.mss_loss import MSSLoss5 as MSSLoss ## refactor !
from loss.mss_loss import DecayCompensator
from loss.reg import FreqSampledIRReg as FreqSampledIRReg

from metric.metric_new_2 import difference_profile, LogEDRDifference
from metric.plot_edr import compare_edr
from metric.plot_training_status import *
from components.biquad_complex import Biquad
from utils import rms_normalize

from loss import MultiScaleSpectralLoss

class FVNEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(**vars(args))
        self.fvn_dec = FVN_Decoder(input_shape=input_shape, 
                                   num_seg=20,
                                   gain_per_seg=4,
                                   fir_len=100)
        self.svf_dec = SVF_Decoder(input_shape=input_shape, 
                                   num_channel=20,
                                   svf_per_channel=8,
                                   parallel=False)
        self.fvn = FVN(seg_num=20, 
                       gain_per_seg=4, 
                       ir_length=120000, 
                       nonuniform_segmentation=True,
                       num_sap=20, 
                       cumulative_sap=True)


    def forward(self, z):
        params_fvn = self.fvn_dec(z)
        params_svf = self.svf_dec(z)
        color = self.svf(params_svf)
        color = util.prod(color.transpose(-1, -2))
        fvn_ir = self.fvn(params_fvn, color) 
        fvn_ir = fvn_ir[:, :120000]
        return fvn_ir

class AFVNEstimator(nn.Module):
    def __init__(self, args):
        self.__dict__.update(**vars(args))

class ARPEstimationSolver(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(**vars(args))
        self.save_hyperparameters()
        self.encoder = None
        self.decoder = None
        self.loss = LossHandler(['spectral', 'decay', 'parameter'])
        self.metric = MetricHandler()

    def forward(self, reference):
        z = self.encoder(reference)
        ar_p = self.ar_proj(z)


    def training_step(self, batch, idx):
        reference = batch['reference']
        pred_ir = self.forward(reference)
        prototype_loss, prototype_loss_dict = self.prototype_loss(pred, G_int)
        self.log_dict({**prototype_loss_dict, **parameter_loss_dict})
        return prototype_loss+parameter_loss

    @torch.no_grad()
    def validation_step(self, batch, idx, loader_idx):
        G, G_int = batch['G'], batch['G_intermediate']
        pred = self.autoencoding_forward(G, G_int)
        prototype_loss, prototype_loss_dict = self.prototype_loss(pred, G_int, prefix='valid')
        parameter_loss, parameter_loss_dict = self.parameter_loss(pred, G, prefix='valid')
        self.log_dict({**prototype_loss_dict, **parameter_loss_dict})
        self.log_dict(self.metric.get_decoding_step_metric(pred, G_int, prefix='valid'))

        if self.current_epoch != 0 and self.current_epoch%10 == 0:
            if idx < 3:
                embeddings = self.g_embedder(G, embed_parameter=True, embed_global=True, task=2)
                z_g = self.g_encoder(embeddings, heads=['global'])['global']
                if self.zero_latent: z_g = z_g*0.
                subset = 'seen-source-distribution' if loader_idx == 0 else 'unseen-source-distribution'
                save_dir = opj(self.save_dir, 'valid', subset, str(idx))
                self.decode_full_inference(z_g, batch, prototype=False, plot=idx<3, save=idx<3, save_dir=save_dir)
                self.decode_parameter_inference(z_g, batch, plot=idx<3, save=idx<3, save_dir=save_dir)
        return prototype_loss+parameter_loss

    @torch.no_grad()
    def test_step(self, batch, idx, loader_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def log_dict(self, log_dict):
        super().log_dict(log_dict, prog_bar=True, logger=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
