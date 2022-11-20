import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset 
import torch.autograd.profiler as profiler

import numpy as np
import soundfile as sf

from components import core_complex as comp 
from components import core_dsp_complex as dsp
from components import filter_complex as filt
from components import utils as util

from components.fvn_complex_3 import FVN2, TrueFVN
from components.svf import SVF
from components.biquad_complex import Biquad
from train_reverb.network.fvn_decoder import FVN_Decoder as FVN_Decoder
from train_reverb.network.svf_decoder import SVF_Decoder as SVF_Decoder
from train_reverb.network.svf_decoder import SVFParametricEQ_Decoder_Stable_Shelving3 as SVFParametricEQ_Decoder
from train_reverb.network.biq_decoder import Biquad_Decoder as Biquad_Decoder
from train_reverb.network.fir_decoder import FIR_LinPhase_Decoder
from train_reverb.network.encoder_rnn_optimized_2 import Encoder
from train_reverb.loss.mss_loss import MSSLoss5 as MSSLoss ## refactor !
from train_reverb.loss.mss_loss import DecayCompensator
from train_reverb.loss.reg import FreqSampledIRReg as FreqSampledIRReg
from train_reverb.loss.paramloss import ParameterMatchLossFVN2 as ParameterMatchLoss

from train_reverb.metric.metric_new_2 import difference_profile 
from train_reverb.metric.plot_edr import compare_edr
from train_reverb.metric.plot_training_status import *
from components.biquad_complex import Biquad
from utils import rms_normalize

from tqdm import tqdm
import pickle

from datetime import datetime
import os, sys
import random

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mturk', dest = 'mturk', action = 'store_true', help = 'set regularization')
parser.set_defaults(mturk = False)

parser.add_argument('--profile', dest = 'profile', action = 'store_true', help = 'set regularization')
parser.set_defaults(profile = False)

parser.add_argument('--debug', dest = 'debug', action = 'store_true', help = 'enable debug')
parser.set_defaults(debug = False)

parser.add_argument('--tspeech_input', dest = 'blind_estimation', action = 'store_true', help = 'set regularization')
parser.add_argument('--rir_input', dest = 'blind_estimation', action = 'store_false', help = 'disable regularization')
parser.set_defaults(blind_estimation = True)

parser.add_argument('--both_input', dest = 'mixed_estimation', action = 'store_true', help = 'disable regularization')
parser.set_defaults(mixed_estimation = False)

parser.add_argument('--epochs', type = int, default = 10, help = 'run epochs')
parser.add_argument('--batch_size', type = int, default = 4, help = 'batch size')
parser.add_argument('--seed', type = int, default = 121, help = 'seed for data split (valid & test)')

parser.add_argument('--speech_len', type = int, default = 120000, help = 'speech length (in sample) that model will see')
parser.add_argument('--ir_len', type = int, default = 120000, help = 'ir length (in sample) that model will generate')
parser.add_argument('--sr', type = int, default = 48000, help = 'sample rate')
parser.add_argument('--num_workers', type = int, default = 4, help = 'num worker for dataloader')

parser.add_argument('--ir_eq', dest = 'ir_eq', action = 'store_true', help = 'random eq-ing')
parser.set_defaults(ir_eq = False)

parser.add_argument('--ir_eq_range', type = int, nargs = '+', default = [2, 5], help = 'random eq-ing range')

parser.add_argument('--fvn_mode', type = str, default = 'basic', help = 'mode of fvn - basic or advanced')

parser.add_argument('--num_seg', type = int, default = 80, help = 'num of segment that FVN will generate')
parser.add_argument('--nonuniform_segmentation', dest = 'nonuniform_segmentation', action = 'store_true', help = 'set non-uniform segmentation')
parser.set_defaults(nonuniform_segmentation = False)

parser.add_argument('--gain_per_seg', type = int, default = 2, help = 'subsegments for gain, gain per each segment')

parser.add_argument('--pulse_distance', type = int, default = 20, help = 'pulse distance (global)') # todo !! change pulse distance !!
parser.add_argument('--delta_pulse_distance', type = int, default = 10, help = 'pulse distance (global)') # todo !! change pulse distance !!
parser.add_argument('--nonuniform_pulse_distance', dest = 'nonuniform_pulse_distance', action = 'store_true', help = 'set non-uniform pulse distance')
parser.set_defaults(nonuniform_pulse_distance = False)

parser.add_argument('--num_sap', type = int, default = 20, help = 'num of segment that FVN will generate')
parser.add_argument('--cumulative_sap', dest = 'cumulative_sap', action = 'store_true', help = 'set cumulative allpass filter')
parser.set_defaults(cumulative_sap = False)

parser.add_argument('--filter', type = str, default = 'svf_parallel', help = 'filter to use')
parser.add_argument('--param_per_seg', type = int, default = 80, help = 'num parameters for fully-filtered fvn segments (all segments for basic fvn, only first segment for advanced fvm)')
parser.add_argument('--param_per_delta_seg', type = int, default = 5, help = 'num parameters for fvn segments with delta coloration filters (advanced fvn only)')
parser.add_argument('--iir_sample', type = int, default = 4000, help = 'length of frequency-sampled filters')

parser.add_argument('--no_fir', dest = 'fir', action = 'store_false', help = 'enable fir')
parser.set_defaults(fir = True)

parser.add_argument('--encoder_spec', type = str, default = 'log', help = 'frequency-axis scale of encoder spectrogram - lin, mel, log or cqt')

parser.add_argument('--weighted_loss', type = bool, default = False, help = 'weighted loss for decay compensation')
parser.add_argument('--loss_spec', type = str, default = 'log', help = 'frequency-axis scale of spectrogram loss - lin, mel, log or cqt')
parser.add_argument('--loss_nfft', type = int, nargs = '+', default = [256, 512, 1024, 2048, 4096], help = 'nfft of MSSLoss') # more detailed? 8192?
parser.add_argument('--loss_alpha', type = float, default = 0., help = 'alpha (weight of log loss term) of MSSLoss')
parser.add_argument('--lr', type = float, default = 1e-5, help = 'initial lr of optimizer')

parser.add_argument('--set_regularization', dest = 'regularization', action = 'store_true', help = 'set regularization')
parser.add_argument('--no_regularization', dest = 'regularization', action = 'store_false', help = 'disable regularization')
parser.set_defaults(regularization = True)

parser.add_argument('--print_loss_per', type = int, default = 10000, help = 'print out average of recent losses')
parser.add_argument('--plot_status_per', type = int, default = 10000, help = 'plot and save training status')
parser.add_argument('--checkpoint', type = str, default = None, help = 'checkpoint of saved model')
parser.add_argument('--num_sample_train_jndr', type = int, default = 200, help = 'num of samples for calculating jndr at train set')

parser.add_argument('--reg_view_range', type = int, default = 500, help = 'view range')

args = parser.parse_args()

if args.mturk:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat import Generate_Dataset_Param_Baseline
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat_mturk import Generate_Dataset
else:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat import Generate_Dataset, Generate_Dataset_Param_Baseline

parallel = True if (args.filter == 'svf_parallel' or args.filter == 'biquad_parallel') else False
advanced = False if args.fvn_mode == 'basic' else True

cont = False
if args.checkpoint is not None:
    cont = True
    ckpt = torch.load(args.checkpoint)

nowstr = args.checkpoint[-26:-7] if cont else datetime.now().strftime('%Y_%m_%d_%H_%M_%S/')

random.seed(args.seed)

""" DATA GENERATION """
_, vali_set, test_set = Generate_Dataset(ir_len = args.ir_len / args.sr, rdeq = args.ir_eq, rdeq_range = args.ir_eq_range, sr = args.sr, vali_dspeech_idx = True, noiseir_ratio = 0.) # ? speech length?
train_set = Generate_Dataset_Param_Baseline(ir_len = args.ir_len / args.sr, rdeq = args.ir_eq, rdeq_range = args.ir_eq_range, sr = args.sr, vali_dspeech_idx = True, noiseir_ratio = 0.)
print(len(train_set) // 4, len(vali_set) // 4, len(test_set) // 4)
vali_speech_datasize = len(vali_set.speech_directory_list)
test_speech_datasize = len(test_set.speech_directory_list)
test_mdspeech_data_idxes = list(np.random.randint(test_speech_datasize, size = len(test_set)))
print(test_mdspeech_data_idxes)
train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers, worker_init_fn = lambda _: np.random.seed())
vali_loader = DataLoader(vali_set, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers)
test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers)

""" NETWORK """
enc = Encoder(sr = args.sr, nfft = 1024).cuda()

if args.blind_estimation:
    #input_shape = (133, 256)
    input_shape = (109, 256)
else:
    input_shape = (109, 256)

fvn_dec = FVN_Decoder(input_shape = input_shape, num_seg = args.num_seg, gain_per_seg = args.gain_per_seg, fir_len = 50 if args.fir else 1).cuda()
fvn = FVN2(seg_num = args.num_seg, 
           gain_per_seg = args.gain_per_seg, 
           ir_length = args.ir_len,
           seg_length = args.ir_len // args.num_seg, 
           nonuniform_segmentation = args.nonuniform_segmentation,
           pulse_distance = args.pulse_distance, 
           delta_pulse_distance = args.delta_pulse_distance,
           nonuniform_pulse_distance = args.nonuniform_pulse_distance,
           num_sap = args.num_sap,
           cumulative_sap = args.cumulative_sap,
           apply_additional_gain = True if (args.fvn_mode == 'advanced' and args.param_per_delta_seg == 0) else False,
           device = 'cuda').cuda()

truefvn = TrueFVN(connection = 'parallel' if args.filter == 'svf_parallel' else 'serial',
                  mode = 'basic' if args.fvn_mode == 'basic' else 'advanced',
                  seg_num = args.num_seg, 
                  gain_per_seg = args.gain_per_seg, 
                  ir_length = args.ir_len,
                  seg_length = args.ir_len // args.num_seg, 
                  init_num_svf = args.param_per_seg // 5,
                  delta_num_svf = args.param_per_delta_seg // 5,
                  nonuniform_segmentation = args.nonuniform_segmentation,
                  pulse_distance = args.pulse_distance, 
                  delta_pulse_distance = args.delta_pulse_distance,
                  nonuniform_pulse_distance = args.nonuniform_pulse_distance,
                  num_sap = args.num_sap,
                  cumulative_sap = args.cumulative_sap,
                  apply_additional_gain = True if (args.fvn_mode == 'advanced' and args.param_per_delta_seg == 0) else False,
                  device = 'cpu')

if advanced:
    if args.filter == 'svf_parallel' or args.filter == 'svf_serial':
        init_num_svf, delta_num_svf = args.param_per_seg // 5, args.param_per_delta_seg // 5
        if init_num_svf != 0 or delta_num_svf != 0:
            svf_per_channel = init_num_svf + (args.num_seg - 1) * delta_num_svf
            svf_dec = SVF_Decoder(input_shape = input_shape, 
                                  num_channel = 1, 
                                  svf_per_channel = svf_per_channel, 
                                  parallel = parallel, 
                                  sr = args.sr, 
                                  f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
                                  advanced = True).cuda()
    elif args.filter == 'biquad_parallel' or args.filter == 'biquad_serial':
        init_num_biquad, delta_num_biquad = args.param_per_seg // 5, args.param_per_delta_seg // 5
        biquad_per_channel = init_num_biquad + (args.num_seg - 1) * delta_num_biquad
        biquad_dec = Biquad_Decoder(input_shape = input_shape, 
                                    num_channel = 1, 
                                    biquad_per_channel = biquad_per_channel, 
                                    parallel = parallel,
                                    sr = args.sr,
                                    f_init = (40, 12000) if args.sr == 48000 else (40, 4000)).cuda()
    elif args.filter == 'parametric_eq':
        init_num_svf, delta_num_svf = args.param_per_seg // 5, args.param_per_delta_seg // 5
        svf_per_channel = init_num_svf + (args.num_seg - 1) * delta_num_svf
        svf_dec = SVFParametricEQ_Decoder(input_shape = input_shape,
                                          num_channel = 1,
                                          svf_per_channel = svf_per_channel,
                                          sr = args.sr,
                                          f_init = (40, 12000) if args.sr == 48000 else (40, 4000)).cuda()
else:
    if args.filter == 'svf_parallel' or args.filter == 'svf_serial':
        svf_per_channel = args.param_per_seg // 5
        svf_dec = SVF_Decoder(input_shape = input_shape, 
                              num_channel = args.num_seg, 
                              svf_per_channel = svf_per_channel, 
                              parallel = parallel,
                              sr = args.sr,
                              f_init = (40, 12000) if args.sr == 48000 else (40, 4000)).cuda()
    elif args.filter == 'biquad_parallel' or args.filter == 'biquad_serial':
        biquad_per_channel = args.param_per_seg // 5
        biquad_dec = Biquad_Decoder(input_shape = input_shape, num_channel = args.num_seg, biquad_per_channel = biquad_per_channel, parallel = parallel).cuda()
    elif args.filter == 'parametric_eq':
        svf_per_channel = args.param_per_seg // 5
        svf_dec = SVFParametricEQ_Decoder(input_shape = input_shape,
                                          num_channel = args.num_seg,
                                          svf_per_channel = svf_per_channel,
                                          sr = args.sr,
                                          f_init = (40, 12000) if args.sr == 48000 else (40, 4000)).cuda()
    elif args.filter == 'linphase_fir':
        fir_dec = FIR_LinPhase_Decoder(input_shape = input_shape, 
                                       num_channel = args.num_seg,
                                       fir_len = args.param_per_seg).cuda()

if args.filter == 'svf_parallel' or args.filter == 'svf_serial' or args.filter == 'parametric_eq':
    svf = SVF(num_sample = args.iir_sample).cuda()
if args.filter == 'biquad_serial':
    biquad = Biquad(num_sample = args.iir_sample).cuda()

net_params = list(enc.parameters()) + list(fvn_dec.parameters())
if args.filter in ['svf_parallel', 'svf_serial', 'parametric_eq']:
    if not (args.fvn_mode == 'advanced' and args.param_per_seg == 0 and args.param_per_delta_seg == 0):
        net_params = net_params + list(svf_dec.parameters())
elif args.filter in ['linphase_fir']:
    net_params = net_params + list(fir_dec.parameters())

total_num_params = sum(p.numel() for p in net_params if p.requires_grad)

comp = DecayCompensator().cuda()
mss = MSSLoss(args.loss_nfft, args.loss_spec, alpha = args.loss_alpha, sr = args.sr).cuda()
param_loss = ParameterMatchLoss().cuda()
reg = FreqSampledIRReg(view_range = args.reg_view_range, norm = 'l1', reduce_axis = (-1,), eps = 1e-12)
optim = torch.optim.Adam(params = net_params, lr = args.lr)
if args.blind_estimation:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], gamma = 0.1 ** (0.1), verbose = True)
else:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9], gamma = 0.1 ** (0.2), verbose = True)

#print(enc, fvn_dec, svf_dec, fvn)
print("TOTAL PARAMS", total_num_params)

if cont:
    enc.load_state_dict(ckpt['enc'])
    fvn_dec.load_state_dict(ckpt['fvn_dec'])
    svf_dec.load_state_dict(ckpt['svf_dec'])
    optim.load_state_dict(ckpt['opt'])
    sched.load_state_dict(ckpt['scd'])

if cont:
    save_dir = args.checkpoint[:-18]
else:
    if args.debug:
        save_dir = '/HDD1/IIRFVN4/debug/'
        os.makedirs(save_dir, exist_ok = True)
    else:
        save_dir = '/HDD1/IIRFVN4/' + nowstr
        os.makedirs(save_dir)

os.makedirs(save_dir + 'train/', exist_ok = True)
os.makedirs(save_dir + 'vali/', exist_ok = True)
os.makedirs(save_dir + 'test/', exist_ok = True)
os.makedirs(save_dir + 'checkpoints/', exist_ok = True)

match_losses = []
param_losses = []
if args.regularization:
    reg_losses = []
init_ep = int(args.checkpoint[-5:-3]) + 1 if cont else 0
step = 0

fig, axes = None, None
fig3, axes3 = None, None
for ep in range(init_ep, args.epochs):
    train_dir = save_dir + 'train/' + str(ep).zfill(2) + '/'
    vali_dir = save_dir + 'vali/' + str(ep).zfill(2) + '/'
    os.makedirs(train_dir, exist_ok = True) 
    os.makedirs(vali_dir, exist_ok = True)
    # TRAIN 
    enc.train()
    diff_fulls, diff_avgs = {}, {}
    for i, dspeech in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
                if args.fvn_mode == 'basic':
                    gt_params_svf = svf_dec.generate_params()
                    gt_params_fvn = fvn_dec.generate_params()
                elif args.fvn_mode == 'advanced':
                    gt_params_svf = svf_dec.generate_advfvn_params()
                    gt_params_fvn = fvn_dec.generate_advfvn_params()
                gt_color = svf(gt_params_svf)
                if args.fvn_mode == 'basic':
                    gt_color = util.prod(gt_color.transpose(-1, -2))
                elif args.fvn_mode == 'advanced':
                    gt_color = util.cumprod2(gt_color.transpose(-1, -2)).squeeze(-3)
                    gt_color = gt_color.transpose(-1, -2)[:, init_num_svf - 1::delta_num_svf, :]

            ir = fvn(gt_params_fvn, gt_color) 
            ir = ir[:, :args.ir_len]
            ir_norm = torch.sqrt(torch.sum(ir ** 2, -1, keepdim = True))
            ir = ir / ir_norm
            for key in gt_params_fvn.keys():
                gt_params_fvn[key] = gt_params_fvn[key] / ir_norm.view(4, 1, 1)
            if args.blind_estimation:
                dspeech = dspeech.float().cuda()
                tspeech = dsp.convolve(dspeech, ir)[..., 120000:240000]

        optim.zero_grad()

        if args.blind_estimation:
            latent = enc(tspeech)
        else:
            latent = enc(ir)

        params_fvn = fvn_dec(latent, g_db = True)
        params_svf = svf_dec(latent)

        loss = param_loss({**gt_params_fvn, **gt_params_svf}, {**params_fvn, **params_svf})

        if i % args.plot_status_per == 0:
            with torch.no_grad():
                color = svf(params_svf)
                if args.fvn_mode == 'basic':
                    color = util.prod(color.transpose(-1, -2))
                elif args.fvn_mode == 'advanced':
                    color = util.cumprod2(color.transpose(-1, -2)).squeeze(-3)
                    color = color.transpose(-1, -2)[:, init_num_svf - 1::delta_num_svf, :]
                fvn_ir = fvn(params_fvn, color) 
                fvn_ir = fvn_ir[:, :args.ir_len]

                match_loss = mss(ir, fvn_ir)

                Bs, As = filt.svf(params_svf['twoRs'], params_svf['Gs'], params_svf['c_hps'], params_svf['c_bps'], params_svf['c_lps'])
                tfvn_params = {**params_fvn, **dict(Bs = Bs, As = As)}
                tfvn_ir = truefvn(tfvn_params)
                print("true match", mss(ir, tfvn_ir[:, :ir.shape[-1]].cuda()).detach().item())
                print("freqsamp match", match_loss.detach().item())

        loss.backward()
        optim.step()
        param_losses.append(loss.item())

        if i < args.num_sample_train_jndr:
            diff_full, diff_avg, _, _, _, _, _ = difference_profile(ir.detach().cpu().numpy(), fvn_ir.detach().cpu().numpy(), sr = args.sr)

            for key in diff_full.keys():
                if not key in diff_fulls:
                    diff_fulls[key] = [diff_full[key]]
                    diff_avgs[key] = [diff_avg[key]]
                else:
                    diff_fulls[key].append(diff_full[key])
                    diff_avgs[key].append(diff_avg[key])

        elif i == args.num_sample_train_jndr:
            print("DIFF (train) full & freq")
            for key in diff_avgs.keys():
                print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
                print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))

        if step % args.print_loss_per == 0:
            print(nowstr, ep, i, np.average(param_losses))
            param_loss.print_losses()
            param_losses = []

        if i % args.plot_status_per == 0:# and (ep != 0 or (ep == 0 and i != 0)):
            if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
                params = {**params_fvn, **params_svf}
                params['twoRs'] = params['twoRs'].view(-1, params['twoRs'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                params['Gs'] = params['Gs'].view(-1, params['Gs'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                params['c_hps'] = params['c_hps'].view(-1, params['c_hps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                params['c_bps'] = params['c_bps'].view(-1, params['c_bps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                params['c_lps'] = params['c_lps'].view(-1, params['c_lps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                params['g'] = params['g'].view(-1, params['g'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                #for key in params.keys():
                #    print(key, params[key].shape)

                fig2 = plot_svf_fittings(params, sr = args.sr)
                fig2.savefig(train_dir + 'freqz_' + str(i).zfill(5) + '.pdf')
                plt.close(fig2)

                gt_params = {**gt_params_fvn, **gt_params_svf}
                #for key in gt_params.keys():
                #    print(key, gt_params[key].shape)
                gt_params['twoRs'] = gt_params['twoRs'].view(-1, gt_params['twoRs'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                gt_params['Gs'] = gt_params['Gs'].view(-1, gt_params['Gs'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                gt_params['c_hps'] = gt_params['c_hps'].view(-1, gt_params['c_hps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                gt_params['c_bps'] = gt_params['c_bps'].view(-1, gt_params['c_bps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                gt_params['c_lps'] = gt_params['c_lps'].view(-1, gt_params['c_lps'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                gt_params['g'] = gt_params['g'].view(-1, gt_params['g'].shape[-1]).transpose(-1, -2).clone().detach().cpu().numpy()
                #for key in gt_params.keys():
                #    print(key, gt_params[key].shape)

                fig2 = plot_svf_fittings(gt_params, sr = args.sr)
                fig2.savefig(train_dir + 'gt_freqz_' + str(i).zfill(5) + '.pdf')
                plt.close(fig2)

                fig, axes = plot_svf_params(params, fig, axes, step, sr = args.sr)
                fig.savefig(train_dir + 'params_' + str(i).zfill(5) + '.pdf')

                fig3, axes3 = plot_render_errors(params, args.num_seg, 16000, fig3, axes3, step)
                fig3.savefig(train_dir + 'error_' + str(i).zfill(5) + '.pdf')

        if i % args.plot_status_per == 0:#and (ep != 0 or (ep == 0 and i != 0)):
            fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), fvn_ir.detach().cpu().numpy(), sr = args.sr)
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(train_dir + 'edr_' + str(i).zfill(5) + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)
            
        if i % args.plot_status_per == 0:
            for j in range(args.batch_size):
                sf.write(train_dir + 'rir_' + str(i).zfill(5) + '_' + str(j) + '.wav', ir[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + 'rec_' + str(i).zfill(5) + '_' + str(j) + '.wav', fvn_ir[j].detach().cpu().numpy(), args.sr)
                if args.blind_estimation:
                    sf.write(train_dir + 'tsp_' + str(i).zfill(5) + '_' + str(j) + '.wav', tspeech[j].detach().cpu().numpy(), args.sr)
                    sf.write(train_dir + 'dsp_' + str(i).zfill(5) + '_' + str(j) + '.wav', dspeech[j].detach().cpu().float().numpy(), args.sr)
                sf.write(train_dir + 'trc_' + str(i).zfill(5) + '_' + str(j) + '.wav', tfvn_ir[j].detach().cpu().float().numpy(), args.sr)

        step = step + 1

        if args.debug:
            if i > 1:
                break

    sched.step()

    # VALID
    enc.eval()
    if ep % 1 == 0:
        if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
            ckpt = dict(enc = enc.state_dict(),
                        fvn_dec = fvn_dec.state_dict(),
                        svf_dec = svf_dec.state_dict(),
                        #fir_dec = fir_dec.state_dict(),
                        opt = optim.state_dict(),
                        scd = sched.state_dict())
        else:
            ckpt = dict(enc = enc.state_dict(),
                        fvn_dec = fvn_dec.state_dict(),
                        #svf_dec = svf_dec.state_dict(),
                        #fir_dec = fir_dec.state_dict(),
                        opt = optim.state_dict(),
                        scd = sched.state_dict())
        torch.save(ckpt, save_dir + 'checkpoints/' + str(ep).zfill(2) + '.pt')

        val_losses = []
        with torch.no_grad():
            diff_fulls, diff_avgs, p_full_rirs, p_full_fvns, p_rirs, p_fvns = {}, {}, {}, {}, {}, {}
            for i, (ir, tspeech, dspeech, dspeech_idx) in enumerate(tqdm(vali_loader)):
                ir = ir / torch.sqrt(torch.sum(ir ** 2, -1, keepdim = True))

                if not args.mixed_estimation:
                    if args.blind_estimation:
                        tspeech = tspeech.float().cuda()
                        ir = ir.float().cuda()
                        latent = enc(tspeech)
                    else:
                        ir = ir.float().cuda()
                        latent = enc(ir)
                else:
                    tspeech = tspeech.float()
                    ir = ir.float()
                    random = (torch.rand(4, 1) > 0.5).float()
                    mixed = ir * random + tspeech * (1 - random) 
                    ir = ir.cuda()
                    mixed = mixed.cuda()
                    latent = enc(mixed)

                params_fvn = fvn_dec(latent, g_db = True)

                color = None
                if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
                    if args.filter == 'svf_parallel' or args.filter == 'svf_serial':
                        params_svf = svf_dec(latent)
                        color = svf(params_svf)
                    elif args.filter == 'biquad_parallel' or args.filter == 'biquad_serial':
                        params_biquad = biquad_dec(latent)
                        color = biquad(params_biquad)
                    elif args.filter == 'parametric_eq':
                        params_svf = svf_dec(latent)
                        color = svf(params_svf)
                    elif args.filter == 'linphase_fir':
                        color = fir_dec(latent)

                    if args.fvn_mode == 'basic':
                        if args.filter == 'svf_parallel' or args.filter == 'biquad_parallel':
                            color = torch.sum(color, -2)
                        elif args.filter == 'svf_serial' or args.filter == 'biquad_serial' or args.filter == 'parametric_eq':
                            color = util.prod(color.transpose(-1, -2))

                    elif args.fvn_mode == 'advanced':
                        if args.filter == 'svf_parallel' or args.filter == 'biquad_parallel':
                            color = torch.fft.irfft(color).squeeze(-3)
                            color = torch.cumsum(color, -2)
                            if delta_num_svf == 0:
                                color = color[:, init_num_svf - 1, :].unsqueeze(-2)
                            else:
                                color = color[:, init_num_svf - 1::delta_num_svf, :]
                            color = torch.fft.rfft(color)
                        elif args.filter == 'svf_serial' or args.filter == 'biquad_serial' or args.filter == 'parametric_eq':
                            color = util.cumprod2(color.transpose(-1, -2)).squeeze(-3)
                            if delta_num_svf == 0:
                                color = color.transpose(-1, -2)[:, init_num_svf - 1, :].unsqueeze(-2)
                            else:
                                if init_num_svf == 0:
                                    color = color.transpose(-1, -2)[:, 0::delta_num_svf, :]
                                    color = torch.cat([torch.ones_like(color[:, :1, :]), color], -2)
                                else:
                                    color = color.transpose(-1, -2)[:, init_num_svf - 1::delta_num_svf, :]

                fvn_ir = fvn(params_fvn, color) 
                fvn_ir = fvn_ir[:, :ir.shape[-1]]

                loss = mss(ir, fvn_ir)
                val_losses.append(loss.detach().item())

                diff_full, diff_avg, _, p_full_rir, p_full_fvn, p_rir, p_fvn = difference_profile(ir.detach().cpu().numpy(), fvn_ir.detach().cpu().numpy(), sr = args.sr)

                for key in diff_full.keys():
                    if not key in diff_fulls:
                        diff_fulls[key] = [diff_full[key]]
                        diff_avgs[key] = [diff_avg[key]]
                        p_full_rirs[key] = [p_full_rir[key]]
                        p_full_fvns[key] = [p_full_fvn[key]]
                        p_rirs[key] = [p_rir[key]]
                        p_fvns[key] = [p_fvn[key]]
                    else:
                        diff_fulls[key].append(diff_full[key])
                        diff_avgs[key].append(diff_avg[key])
                        p_rirs[key].append(p_rir[key])
                        p_fvns[key].append(p_fvn[key])
                        p_full_rirs[key].append(p_full_rir[key])
                        p_full_fvns[key].append(p_full_fvn[key])

                if i < 10:
                    fvn_ir = fvn_ir.detach().cpu().numpy()
                    for j in range(args.batch_size):
                        gt_ir, dspeech, tspeech = vali_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
                        cspeech = signal.convolve(dspeech, fvn_ir[j])

                        mdspeech_idx = np.random.randint(0, vali_speech_datasize)
                        _, mdspeech, _ = vali_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
                        mspeech = signal.convolve(mdspeech, fvn_ir[j])

                        sf.write(vali_dir + str(i) + '_' + str(j) + '_rir.wav', gt_ir, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_transmitted_speech.wav', tspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_dry_speech.wav', dspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_estimation.wav', fvn_ir[j], args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_convolved_speech.wav', cspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_matching_dry_speech.wav', mdspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_matched_speech.wav', mspeech, args.sr)

                    fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), fvn_ir, sr = args.sr)
                    fig_edr.set_size_inches(20, 12)
                    fig_edr.savefig(vali_dir + 'edr_' + str(i) + '.pdf', bbox_inches = 'tight')
                    plt.close(fig_edr)

                if args.debug:
                    if i > 1:
                        break
                    
        valid_summary = {}
        valid_summary['loss'] = np.average(val_losses)
        valid_summary['diff_full'] = diff_fulls
        valid_summary['diff_avg'] = diff_avgs
        valid_summary['p_full_rir'] = p_full_rirs
        valid_summary['p_full_rec'] = p_full_fvns
        valid_summary['p_rir'] = p_rirs
        valid_summary['p_rec'] = p_fvns
        valid_summary['rir_dir'] = vali_set.ir_directory_list

        print('epoch:', ep, "loss:", np.average(val_losses))
        print('DIFF')

        for key in diff_avgs.keys():
            print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
            print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))


        with open(vali_dir + 'result.pickle', 'wb') as fp:
            pickle.dump(valid_summary, fp)

""" TEST """ 
test_losses, true_losses = [], []

if args.blind_estimation:
    task_str = 'be_'
else:
    task_str = 'as_'
with torch.no_grad():
    diff_fulls, diff_avgs, p_full_rirs, p_full_fvns, p_rirs, p_fvns = {}, {}, {}, {}, {}, {}
    tdiff_fulls, tdiff_avgs, p_full_tfvns, p_tfvns = {}, {}, {}, {}
    for i, (ir, tspeech, dspeech, dspeech_idx) in enumerate(tqdm(test_loader)):
        ir = ir / torch.sqrt(torch.sum(ir ** 2, -1, keepdim = True))
        if args.mturk:
            test_dir = save_dir + 'mturk/' + str(i).zfill(3) + '/'
        else:
            test_dir = save_dir + 'test/' + str(i).zfill(3) + '/'
        os.makedirs(test_dir, exist_ok = True)

        if not args.mixed_estimation:
            if args.blind_estimation:
                tspeech = tspeech.float().cuda()
                ir = ir.float().cuda()
                latent = enc(tspeech)
            else:
                ir = ir.float().cuda()
                latent = enc(ir)
        else:
            tspeech = tspeech.float()
            ir = ir.float()
            random = (torch.rand(4, 1) > 0.5).float()
            mixed = ir * random + tspeech * (1 - random) 
            ir = ir.cuda()
            mixed = mixed.cuda()
            latent = enc(mixed)

        params_fvn = fvn_dec(latent, g_db = True)

        color = None
        if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
            if args.filter == 'svf_parallel' or args.filter == 'svf_serial':
                params_svf = svf_dec(latent)
                color = svf(params_svf)
            elif args.filter == 'biquad_parallel' or args.filter == 'biquad_serial':
                params_biquad = biquad_dec(latent)
                color = biquad(params_biquad)
            elif args.filter == 'parametric_eq':
                params_svf = svf_dec(latent)
                color = svf(params_svf)
            elif args.filter == 'linphase_fir':
                color = fir_dec(latent)

            if args.fvn_mode == 'basic':
                if args.filter == 'svf_parallel' or args.filter == 'biquad_parallel':
                    color = torch.sum(color, -2)
                elif args.filter == 'svf_serial' or args.filter == 'biquad_serial' or args.filter == 'parametric_eq':
                    color = util.prod(color.transpose(-1, -2))

            elif args.fvn_mode == 'advanced':
                if args.filter == 'svf_parallel' or args.filter == 'biquad_parallel':
                    color = torch.fft.irfft(color).squeeze(-3)
                    color = torch.cumsum(color, -2)
                    if delta_num_svf == 0:
                        color = color[:, init_num_svf - 1, :].unsqueeze(-2)
                    else:
                        color = color[:, init_num_svf - 1::delta_num_svf, :]
                    color = torch.fft.rfft(color)
                elif args.filter == 'svf_serial' or args.filter == 'biquad_serial' or args.filter == 'parametric_eq':
                    color = util.cumprod2(color.transpose(-1, -2)).squeeze(-3)
                    if delta_num_svf == 0:
                        color = color.transpose(-1, -2)[:, init_num_svf - 1, :].unsqueeze(-2)
                    else:
                        if init_num_svf == 0:
                            color = color.transpose(-1, -2)[:, 0::delta_num_svf, :]
                            color = torch.cat([torch.ones_like(color[:, :1, :]), color], -2)
                        else:
                            color = color.transpose(-1, -2)[:, init_num_svf - 1::delta_num_svf, :]

        fvn_ir = fvn(params_fvn, color) 
        fvn_ir = fvn_ir[:, :ir.shape[-1]]

        if args.filter in ['svf_serial', 'parametric_eq', 'svf_parallel']:
            if args.fvn_mode == 'basic' or (args.fvn_mode == 'advanced' and (init_num_svf != 0 or delta_num_svf != 0)):
                Bs, As = filt.svf(params_svf['twoRs'], params_svf['Gs'], params_svf['c_hps'], params_svf['c_bps'], params_svf['c_lps'])
            else:
                Bs, As = None, None
            tfvn_params = {**params_fvn, **dict(Bs = Bs, As = As)}
            tfvn_ir = truefvn(tfvn_params)[:, :ir.shape[-1]]
            true_losses.append(mss(ir, tfvn_ir.cuda()).detach().item())
        else:
            tfvn_ir = fvn_ir
            true_losses.append(mss(ir, tfvn_ir.cuda()).detach().item())

        loss = mss(ir, fvn_ir)
        test_losses.append(loss.detach().item())

        diff_full, diff_avg, _, p_full_rir, p_full_fvn, p_rir, p_fvn = difference_profile(ir.detach().cpu().numpy(), fvn_ir.detach().cpu().numpy(), sr = args.sr)
        tdiff_full, tdiff_avg, _, _, p_full_tfvn, _, p_tfvn = difference_profile(ir.detach().cpu().numpy(), tfvn_ir.detach().cpu().numpy(), sr = args.sr)

        for key in diff_full.keys():
            if not key in diff_fulls:
                diff_fulls[key] = [diff_full[key]]
                diff_avgs[key] = [diff_avg[key]]
                p_full_rirs[key] = [p_full_rir[key]]
                p_full_fvns[key] = [p_full_fvn[key]]
                p_rirs[key] = [p_rir[key]]
                p_fvns[key] = [p_fvn[key]]
                tdiff_fulls[key] = [tdiff_full[key]]
                tdiff_avgs[key] = [tdiff_avg[key]]
                p_full_tfvns[key] = [p_full_tfvn[key]]
                p_tfvns[key] = [p_tfvn[key]]
            else:
                diff_fulls[key].append(diff_full[key])
                diff_avgs[key].append(diff_avg[key])
                p_rirs[key].append(p_rir[key])
                p_fvns[key].append(p_fvn[key])
                p_full_rirs[key].append(p_full_rir[key])
                p_full_fvns[key].append(p_full_fvn[key])
                tdiff_fulls[key].append(tdiff_full[key])
                tdiff_avgs[key].append(tdiff_avg[key])
                p_full_tfvns[key].append(p_full_tfvn[key])
                p_tfvns[key].append(p_tfvn[key])

        fvn_ir = fvn_ir.detach().cpu().numpy()
        tfvn_ir = tfvn_ir.detach().cpu().numpy()
        for j in range(args.batch_size):
            gt_ir, dspeech, tspeech = test_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
            cspeech = signal.convolve(dspeech, fvn_ir[j])
            tcspeech = signal.convolve(dspeech, tfvn_ir[j])

            mdspeech_idx = test_mdspeech_data_idxes.pop()
            _, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
            mspeech = signal.convolve(mdspeech, fvn_ir[j])
            tmspeech = signal.convolve(mdspeech, tfvn_ir[j])

            _, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i)
            sf.write(test_dir + task_str + str(j) + '_rir.wav', gt_ir, args.sr)
            sf.write(test_dir + task_str + str(j) + '_rir_crop.wav', ir[j][:args.iir_sample].detach().cpu().numpy(), args.sr)
            sf.write(test_dir + task_str + str(j) + '_transmitted_speech.wav', rms_normalize(tspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_dry_speech.wav', rms_normalize(dspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_matching_dry_speech.wav', rms_normalize(mdspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_estimation.wav', fvn_ir[j], args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_convolved_speech.wav', rms_normalize(cspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_matched_speech.wav', rms_normalize(mspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_true_estimation.wav', tfvn_ir[j], args.sr)
            sf.write(test_dir + task_str + str(j) + '_true_convolved_speech.wav', rms_normalize(tcspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_true_matched_speech.wav', rms_normalize(tmspeech), args.sr)

        fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), fvn_ir, sr = args.sr)
        fig_edr.set_size_inches(20, 12)
        fig_edr.savefig(test_dir + 'edr' + '.pdf', bbox_inches = 'tight')
        plt.close(fig_edr)

        fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), tfvn_ir, sr = args.sr)
        fig_edr.set_size_inches(20, 12)
        fig_edr.savefig(test_dir + 'tedr' + '.pdf', bbox_inches = 'tight')
        plt.close(fig_edr)

        total_params = {**params_fvn, **params_svf}
        for key in total_params.keys():
            total_params[key] = total_params[key].clone().detach().cpu().numpy()

        pickle.dump(total_params, open(test_dir + 'param.pickle', 'wb'))

        if args.debug:
            if i > 1:
                break
            
test_summary = {}
test_summary['loss'] = (np.average(test_losses), test_losses)
test_summary['true_loss'] = (np.average(true_losses), true_losses)
test_summary['diff_full'] = diff_fulls
test_summary['diff_avg'] = diff_avgs
test_summary['true_diff_full'] = tdiff_fulls
test_summary['true_diff_avg'] = tdiff_avgs
test_summary['p_full_rir'] = p_full_rirs
test_summary['p_full_rec'] = p_full_fvns
test_summary['p_full_true_rec'] = p_full_tfvns
test_summary['p_rir'] = p_rirs
test_summary['p_recn'] = p_fvns
test_summary['p_true_rec'] = p_tfvns
test_summary['rir_dir'] = test_set.ir_directory_list

print('final:', "loss:", np.average(test_losses))
print('final:', "true loss:", np.average(true_losses))
print('DIFF')

for key in diff_avgs.keys():
    print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
    print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))
    print(key, '(true mean) %.3f' % np.average(np.concatenate(tdiff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(tdiff_avgs[key], 0)))
    print(key, '(true median) %.3f' % np.median(np.concatenate(tdiff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(tdiff_avgs[key], 0)))

with open(save_dir + 'test/result.pickle', 'wb') as fp:
    pickle.dump(test_summary, fp)
