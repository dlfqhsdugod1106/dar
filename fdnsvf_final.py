import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset 
import torch.autograd.profiler as profiler
from nnAudio import features

import numpy as np
import soundfile as sf

from components import core_complex as comp 
from components import core_dsp as dsp
from components import filter_complex as filt

from components.fdn_complex import FDN, TrueFDN, TrueFDN2
from components.svf import SVF, SVFBell, SVFParametricEQ
from components.sap import SAP
from train_reverb.network.fdn_decoder import FDN_Decoder
from train_reverb.network.svf_decoder import SVF_Decoder, SVFBell_Decoder, SVFParametricEQ_Decoder, SVFParametricEQ_Decoder_Stable_Shelving2, SVFParametricEQ_Decoder_Stable_Shelving3, SVFParametricEQ_Decoder_Stable_Shelving2_Gumbel_Softmax, SVF_Decoder222
from train_reverb.network.biq_decoder import Biquad_Decoder as Biquad_Decoder
from train_reverb.network.sap_decoder import SAP_Decoder
from train_reverb.network.encoder_rnn_optimized_2 import Encoder
from train_reverb.loss.mss_loss import MSSLoss5 as MSSLoss ## refactor !
from train_reverb.loss.mss_loss import DecayCompensator
from train_reverb.loss.reg import FreqSampledIRReg as FreqSampledIRReg

from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat import Generate_Dataset
from train_reverb.metric.metric_new_2 import difference_profile 
from train_reverb.metric.plot_edr import compare_edr
from train_reverb.metric.plot_fdn import plot_filter_params, plot_filter_responses
from train_reverb.metric.plot_training_status import *
from components.biquad_complex import Biquad
import components.utils as util
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

parser.add_argument('--time_alias', dest = 'time_alias', action = 'store_true', help = 'enable time-aliasing for gt ir')
parser.set_defaults(time_alias = False)

parser.add_argument('--num_channel', type = int, default = 8, help = 'num channel of FDN')

parser.add_argument('--filter', type = str, default = 'svf_parallel', help = 'filter to use')
parser.add_argument('--pre_param_per_channel', type = int, default = 40, help = 'num of parameters assigned for pre-filter for each channel')
parser.add_argument('--post_param_per_channel', type = int, default = 40, help = 'num of parameters assigned for post-filter for each channel')
parser.add_argument('--feedback_param_per_channel', type = int, default = 40, help = 'num of parameters assigned for feedback-filter for each channel')

parser.add_argument('--num_sap_per_channel', type = int, default = 4, help = 'number of sap per channel')

parser.add_argument('--same_pre_filter', dest = 'same_pre_filter', action = 'store_true', help = 'set same pre-filter')
parser.set_defaults(same_pre_filter = False)
parser.add_argument('--same_post_filter', dest = 'same_post_filter', action = 'store_true', help = 'set same post-filter')
parser.set_defaults(same_post_filter = False)
parser.add_argument('--same_feedback_filter', dest = 'same_feedback_filter', action = 'store_true', help = 'set same feedback-filter')
parser.set_defaults(same_feedback_filter = False)

parser.add_argument('--trainable_admittance', dest = 'trainable_admittance', action = 'store_true', help = 'trainable addmitance')
parser.set_defaults(trainable_admittance = False)
parser.add_argument('--ds_compensation', dest = 'ds_compensation', action = 'store_true', help = 'ds_compensation')
parser.set_defaults(ds_compensation = False)

parser.add_argument('--iir_sample', type = int, default = 120000, help = 'length of frequency-sampled filters')

parser.add_argument('--encoder_spec', type = str, default = 'log', help = 'frequency-axis scale of encoder spectrogram - lin, mel, log or cqt')

parser.add_argument('--weighted_loss', type = bool, default = False, help = 'weighted loss for decay compensation')
parser.add_argument('--loss_spec', type = str, default = 'log', help = 'frequency-axis scale of spectrogram loss - lin, mel, log or cqt')
parser.add_argument('--loss_nfft', type = int, nargs = '+', default = [256, 512, 1024, 2048, 4096], help = 'nfft of MSSLoss') # more detailed? 8192?
parser.add_argument('--loss_alpha', type = float, default = 0., help = 'alpha (weight of log loss term) of MSSLoss')
parser.add_argument('--lr', type = float, default = 1e-5, help = 'initial lr of optimizer')

parser.add_argument('--no_regularization', dest = 'regularization', action = 'store_false', help = 'disable regularization')
parser.set_defaults(regularization = True)

parser.add_argument('--print_loss_per', type = int, default = 5000, help = 'print out average of recent losses')
parser.add_argument('--plot_status_per', type = int, default = 5000, help = 'plot and save training status')
parser.add_argument('--checkpoint', type = str, default = None, help = 'checkpoint of saved model')
parser.add_argument('--num_sample_train_jndr', type = int, default = 200, help = 'num of samples for calculating jndr at train set')

args = parser.parse_args()
print(args.num_sap_per_channel)

if args.mturk:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat_mturk import Generate_Dataset
else:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat import Generate_Dataset

parallel = True if (args.filter == 'svf_parallel' or args.filter == 'biquad_parallel') else False

cont = False
if args.checkpoint is not None:
    cont = True
    ckpt = torch.load(args.checkpoint)

nowstr = args.checkpoint[-26:-7] if cont else datetime.now().strftime('%Y_%m_%d_%H_%M_%S/')

random.seed(args.seed)

""" DATA GENERATION """
train_set, vali_set, test_set = Generate_Dataset(ir_len = args.ir_len / args.sr, rdeq = args.ir_eq, rdeq_range = args.ir_eq_range, sr = args.sr, vali_dspeech_idx = True, noiseir_ratio = 0., time_alias = args.time_alias) # ? speech length?
vali_speech_datasize = len(vali_set.speech_directory_list)
test_speech_datasize = len(test_set.speech_directory_list)
train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers, worker_init_fn = lambda _: np.random.seed())
vali_loader = DataLoader(vali_set, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers)
test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers)

""" NETWORK """
enc = Encoder(sr = args.sr, nfft = 1024).cuda()

if args.blind_estimation:
    input_shape = (109, 256)
else:
    input_shape = (109, 256)

fdn_dec = FDN_Decoder(input_shape = input_shape, 
                      num_channel = args.num_channel, 
                      #max_ds_len = 6000, 
                      fir_len = 100,
                      trainable_admittance = args.trainable_admittance).cuda()

net_params = []
if args.filter == 'svf_parallel' or args.filter == 'svf_serial':
    if args.pre_param_per_channel != 0:
        svf_pre_dec = SVF_Decoder(input_shape = input_shape, 
                                  num_channel = 1 if args.same_pre_filter else args.num_channel, 
                                  parallel = parallel,
                                  sr = args.sr,
                                  f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
                                  svf_per_channel = args.pre_param_per_channel // 5,
                                  advanced = False).cuda()
        net_params += list(svf_pre_dec.parameters())

    if args.post_param_per_channel != 0:
        svf_post_dec = SVF_Decoder(input_shape = input_shape, 
                                   num_channel = 1 if args.same_post_filter else args.num_channel, 
                                   parallel = parallel,
                                   sr = args.sr,
                                   f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
                                   svf_per_channel = args.post_param_per_channel // 5,
                                   advanced = False).cuda()
        net_params += list(svf_post_dec.parameters())

elif args.filter == 'parametric_eq':
    if args.pre_param_per_channel != 0:
        svf_pre_dec = SVFParametricEQ_Decoder_Stable_Shelving3(input_shape = input_shape, 
                                              num_channel = 1 if args.same_pre_filter else args.num_channel, 
                                              sr = args.sr,
                                              f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
                                              svf_per_channel = args.pre_param_per_channel // 5).cuda()
        net_params += list(svf_pre_dec.parameters())

    if args.post_param_per_channel != 0:
        svf_post_dec = SVFParametricEQ_Decoder_Stable_Shelving3(input_shape = input_shape, 
                                               num_channel = 1 if args.same_post_filter else args.num_channel, 
                                               sr = args.sr,
                                               f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
                                               svf_per_channel = args.post_param_per_channel // 5).cuda()
        net_params += list(svf_post_dec.parameters())


if args.feedback_param_per_channel != 0:
    svf_fb_dec = SVFParametricEQ_Decoder_Stable_Shelving2(input_shape = input_shape, 
                                         num_channel = 1 if args.same_feedback_filter else args.num_channel, 
                                         svf_per_channel = args.feedback_param_per_channel // 5,
                                         sr = args.sr,
                                         f_init = (40, 12000) if args.sr == 48000 else (40, 4000)).cuda()
                                         #parallel = False).cuda()
    #svf_fb_dec = SVF_Decoder222(input_shape = input_shape, 
    #                           num_channel = 1 if args.same_post_filter else args.num_channel, 
    #                           parallel = parallel,
    #                           sr = args.sr,
    #                           f_init = (40, 12000) if args.sr == 48000 else (40, 4000),
    #                           svf_per_channel = args.post_param_per_channel // 5,
    #                           advanced = False).cuda()
    net_params += list(svf_fb_dec.parameters())

    svfbell = SVF(num_sample = args.iir_sample).cuda()

svf = SVF(num_sample = args.iir_sample).cuda()

if args.num_sap_per_channel != 0:
    sap_dec = SAP_Decoder(input_shape, 
                          num_channel = args.num_channel, 
                          sap_per_channel = args.num_sap_per_channel).cuda()
    sap = SAP(num_sample = args.iir_sample).cuda()
    net_params += list(sap_dec.parameters())

net_params += list(enc.parameters()) + list(fdn_dec.parameters()) 

fdn = FDN(num_channel = args.num_channel, num_sample = args.iir_sample).cuda()
tfdn = TrueFDN2(ir_len = args.ir_len, 
               same_pre_filter = args.same_pre_filter, 
               same_post_filter = args.same_post_filter, 
               same_feedback_filter = args.same_feedback_filter,
               parallel = parallel)


total_num_params = sum(p.numel() for p in net_params if p.requires_grad)
""""""""""""""""""""""""""""""
# True FDN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
total_num_params = sum(p.numel() for p in net_params if p.requires_grad)

comp = DecayCompensator().cuda()
mss = MSSLoss(args.loss_nfft, args.loss_spec, alpha = args.loss_alpha, sr = args.sr).cuda()
reg = FreqSampledIRReg(view_range = 500, norm = 'l1', reduce_axis = (-1,), eps = 1e-12)
optim = torch.optim.Adam(params = net_params, lr = args.lr)
if args.blind_estimation:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], gamma = 0.1 ** (0.1), verbose = True)
else:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9], gamma = 0.1 ** (0.2), verbose = True)

#print(enc, fdn_dec, svf_pre_dec, svf_post_dec, svf_fb_dec, fdn)
print("TOTAL PARAMS", total_num_params)

if cont:
    enc.load_state_dict(ckpt['enc'], strict=False)
    fdn_dec.load_state_dict(ckpt['fdn_dec'])
    if args.pre_param_per_channel != 0:
        svf_pre_dec.load_state_dict(ckpt['svf_pre_dec'])
    if args.post_param_per_channel != 0:
        svf_post_dec.load_state_dict(ckpt['svf_post_dec'])
    if args.feedback_param_per_channel != 0:
        svf_fb_dec.load_state_dict(ckpt['svf_fb_dec'])
    if args.num_sap_per_channel != 0:
        sap_dec.load_state_dict(ckpt['sap_dec'])
    #optim.load_state_dict(ckpt['opt'])
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
    """ TRAIN  """
    enc.train()
    diff_fulls, diff_avgs = {}, {}
    for i, (ir, tspeech, dspeech) in enumerate(tqdm(train_loader)):
        #with profiler.profile(with_stack = True, profile_memory = True, use_cuda = True) as prof:
        optim.zero_grad()

        if not args.mixed_estimation:
            if args.blind_estimation:
                tspeech = tspeech.float().cuda()
                ir = ir.float().cuda()
                latent = enc(tspeech)
                #latent = enc(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)))
            else:
                ir = ir.float().cuda()
                latent = enc(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)))
        else:
            tspeech = tspeech.float()
            ir = ir.float()
            random = (torch.rand(4, 1) > 0.5).float()
            mixed = ir * random + tspeech * (1 - random) 
            ir = ir.cuda()
            mixed = mixed.cuda()
            latent = enc(mixed)

        params_fdn = fdn_dec(latent)

        if args.num_sap_per_channel != 0:
            params_sap_fb = sap_dec(latent)
            feedback_sap = util.prod(sap(params_sap_fb).transpose(-1, -2))
        else:
            params_sap_fb = None

        if args.filter == 'svf_parallel' or args.filter == 'svf_serial' or args.filter == 'parametric_eq':
            if args.pre_param_per_channel != 0:
                params_svf_pre = svf_pre_dec(latent)
                if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                    pre = util.prod(svf(params_svf_pre).transpose(-1, -2))
                elif args.filter == 'svf_parallel':
                    pre = torch.sum(svf(params_svf_pre), -2)
                pre = pre.transpose(-1, -2).unsqueeze(-1)
            else:
                params_svf_pre, pre = None, None

            if args.post_param_per_channel != 0:
                params_svf_post = svf_post_dec(latent)
                if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                    post = util.prod(svf(params_svf_post).transpose(-1, -2))
                elif args.filter == 'svf_parallel':
                    post = torch.sum(svf(params_svf_post), -2)
                post = post.transpose(-1, -2).unsqueeze(-2)
            else:
                params_svf_post, post = None, None

            if args.feedback_param_per_channel != 0:
                if args.ds_compensation:
                    params_svf_fb = svf_fb_dec(latent, ds = params_fdn['ds'], sap_params = params_sap_fb)
                else:
                    params_svf_fb = svf_fb_dec(latent)#, ds = params_fdn['ds'])
                fb_filter = util.prod(svfbell(params_svf_fb).transpose(-1, -2))
                if args.num_sap_per_channel != 0:
                    fb_filter = fb_filter * feedback_sap
                fb_filter = fb_filter.transpose(-1, -2)
            else:
                params_svf_fb, fb_filter = None, None
                if args.num_sap_per_channel != 0:
                    fb_filter = feedback_sap.transpose(-1, -2)

            #if args.regularization:
            #    pre_t = torch.fft.irfft(pre)
            #    post_t = torch.fft.irfft(post)
            #    fb_t = torch.fft.irfft(fb_filter)
            #    reg_loss = reg(pre_t) + reg(post_t) + reg(fb_t) * 10


        params_color = dict(pre = pre, post = post, fb_filter = fb_filter)

        fdn_ir = fdn({**params_fdn, **params_color})
        fdn_ir = fdn_ir[:, :args.ir_len]

        if i % args.plot_status_per == 0:
            #print(params_sap_fb['alphas'])
            with torch.no_grad():
                if args.pre_param_per_channel != 0:
                    pre_filter_Bs, pre_filter_As = filt.svf(params_svf_pre['twoRs'], params_svf_pre['Gs'], params_svf_pre['c_hps'], params_svf_pre['c_bps'], params_svf_pre['c_lps'])
                else:
                    ones = torch.ones(args.batch_size, 1, 1)
                    pre_filter_Bs, pre_filter_As = filt.svf(ones, ones, ones, ones, ones)

                if args.post_param_per_channel != 0:
                    post_filter_Bs, post_filter_As = filt.svf(params_svf_post['twoRs'], params_svf_post['Gs'], params_svf_post['c_hps'], params_svf_post['c_bps'], params_svf_post['c_lps'])
                else:
                    ones = torch.ones(args.batch_size, 1, 1)
                    post_filter_Bs, post_filter_As = filt.svf(ones, ones, ones, ones, ones)

                if args.feedback_param_per_channel != 0:
                    fb_filter_Bs, fb_filter_As = filt.svf(params_svf_fb['twoRs'], params_svf_fb['Gs'], params_svf_fb['c_hps'], params_svf_fb['c_bps'], params_svf_fb['c_lps'])
                    print(fb_filter_Bs[0][0], fb_filter_As[0][0])
                else:
                    ones = torch.ones(args.batch_size, 1, 1)
                    fb_filter_Bs, fb_filter_As = filt.svf(ones, ones, ones, ones, ones)

                tfdn_lti_ir, tfdn_ltv_ir = tfdn(ds = params_fdn['ds'].detach().cpu().numpy(),
                                                pre_gain = params_fdn['pre_gain'].detach().cpu().numpy(),
                                                pre_filter_Bs = pre_filter_Bs.detach().cpu().numpy(),
                                                pre_filter_As = pre_filter_As.detach().cpu().numpy(),
                                                fb_adm = params_fdn['fb_adm'].detach().cpu().numpy(),
                                                fb_gain = params_fdn['fb_gain'].detach().cpu().numpy(),
                                                fb_filter_Bs = fb_filter_Bs.detach().cpu().numpy(),
                                                fb_filter_As = fb_filter_As.detach().cpu().numpy(),
                                                #fb_SAP_alphas = params_sap_fb['alphas'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                                #fb_SAP_ms = params_sap_fb['ds'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                                post_gain = params_fdn['post_gain'].detach().cpu().numpy(),
                                                post_filter_Bs = post_filter_Bs.detach().cpu().numpy(),
                                                post_filter_As = post_filter_As.detach().cpu().numpy(),
                                                bypass_FIR = params_fdn['fir'].detach().cpu().numpy())
                
                true_lti_loss = mss(ir, tfdn_lti_ir.float()).detach().item()
                true_ltv_loss = mss(ir, tfdn_ltv_ir.float()).detach().item()
                print("true match", true_lti_loss, true_ltv_loss)

                figtemp, axtemp = plt.subplots(1, 3,sharey = True)
                axtemp[0].plot(tfdn_lti_ir.detach().cpu().numpy()[0][:2000])
                axtemp[1].plot(fdn_ir.detach().cpu().numpy()[0][:2000])
                axtemp[2].plot(fdn_ir.detach().cpu().numpy()[0][:2000] - tfdn_lti_ir.detach().cpu().numpy()[0][:2000])
                figtemp.savefig('wow.pdf')

        if args.weighted_loss:
            ir_comp, fdn_ir_comp = comp(ir, fdn_ir)
            match_loss = mss(ir_comp, fdn_ir_comp)
            if ep != 0 and step % 1000 == 0 and comp.amp_threshold > 1e-3:
                comp.amp_threshold = comp.amp_threshold * (10 ** -.1)
                print("now amp_threshold is", comp.amp_threshold)
        else:
            match_loss = mss(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)), F.pad(fdn_ir, (0, args.ir_len - args.iir_sample)))
            if i % args.plot_status_per == 0:
                print("freqsamp match", match_loss.detach().item())

        match_losses.append(match_loss.detach().item())
        
        if args.regularization:
            loss = match_loss + reg_loss
            reg_losses.append(reg_loss.detach().item())
        else:
            loss = match_loss

        loss.backward()
        nn.utils.clip_grad_norm_(net_params, 10) #################################################################################
        optim.step()

        if i < args.num_sample_train_jndr:
            diff_full, diff_avg, _, _, _, _, _ = difference_profile(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir.detach().cpu().numpy(), sr = args.sr)

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
            if args.regularization:
                print(nowstr, ep, i, np.average(match_losses), np.average(reg_losses))
                match_losses, reg_losses = [], []
            else:
                print(nowstr, ep, i, np.average(match_losses))
                match_losses  = []

        if i % args.plot_status_per == 0: #and (ep != 0 or (ep == 0 and i != 0)):
            fig, axes = plot_filter_params(params_svf_pre, params_svf_post, params_svf_fb)
            fig.savefig(train_dir + 'params_' + str(i).zfill(5) + '.pdf', bbox_inches = 'tight')
            plt.close(fig)

            if not (args.pre_param_per_channel == 0 and args.post_param_per_channel == 0 and args.feedback_param_per_channel == 0):
                figs, axeses = plot_filter_responses(params_fdn, params_svf_pre, params_svf_post, params_svf_fb, sr = args.sr)
                for j in range(len(figs)):
                    figs[j].set_size_inches(3 * args.num_channel, 15)
                    figs[j].savefig(train_dir + 'filters_' + str(i).zfill(5) + '_' + str(j) + '.pdf', bbox_inches = 'tight')
                    plt.close(figs[j])

            fig_edr, axes_edr = compare_edr(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir.detach().cpu().numpy())
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(train_dir + 'edr_' + str(i).zfill(5) + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)

            if args.filter in ['svf_serial', 'parametric_eq', 'svf_parallel']:
                fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), tfdn_ltv_ir.detach().cpu().numpy())
                fig_edr.set_size_inches(20, 12)
                fig_edr.savefig(train_dir + 'tedr_' + str(i).zfill(5) + '.pdf', bbox_inches = 'tight')
                plt.close(fig_edr)
            
        if i % args.plot_status_per == 0:
            for j in range(args.batch_size):
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_rir.wav', ir[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_freq_sampled_estimation.wav', fdn_ir[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_transmitted_speech.wav', tspeech[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_dry_speech.wav', dspeech[j].detach().cpu().float().numpy(), args.sr)
                #sf.write(train_dir + 'trc_' + str(i).zfill(5) + '_' + str(j) + '.wav', fdn_ir[j].detach().cpu().float().numpy(), args.sr) # !!!!!!!!!!!!!!!!
                # wow !! real fdn !! pog
                if args.filter in ['svf_serial', 'parametric_eq', 'svf_parallel']:
                    sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_true_estimation.wav', tfdn_ltv_ir[j].detach().cpu().float().numpy(), args.sr)
                    sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_lti_estimation.wav', tfdn_lti_ir[j].detach().cpu().float().numpy(), args.sr)

        step = step + 1

        if args.debug:
            if i > 1:
                break

    sched.step()

    """ VALID """
    enc.eval()
    if ep % 1 == 0:
        val_losses = []
        with torch.no_grad():
            diff_fulls, diff_avgs, p_full_rirs, p_full_fdns, p_rirs, p_fdns = {}, {}, {}, {}, {}, {}
            for i, (ir, tspeech, dspeech, dspeech_idx) in enumerate(tqdm(vali_loader)):
                if not args.mixed_estimation:
                    if args.blind_estimation:
                        tspeech = tspeech.float().cuda()
                        ir = ir.float().cuda()
                        latent = enc(tspeech)
                    else:
                        ir = ir.float().cuda()
                        latent = enc(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)))
                else:
                    tspeech = tspeech.float()
                    ir = ir.float()
                    random = (torch.rand(4, 1) > 0.5).float()
                    mixed = ir * random + tspeech * (1 - random) 
                    ir = ir.cuda()
                    mixed = mixed.cuda()
                    latent = enc(mixed)

                params_fdn = fdn_dec(latent)

                if args.num_sap_per_channel != 0:
                    params_sap_fb = sap_dec(latent)
                    feedback_sap = util.prod(sap(params_sap_fb).transpose(-1, -2))
                else:
                    params_sap_fb = None

                if args.filter == 'svf_parallel' or args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                    if args.pre_param_per_channel != 0:
                        params_svf_pre = svf_pre_dec(latent)
                        if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                            pre = util.prod(svf(params_svf_pre).transpose(-1, -2))
                        elif args.filter == 'svf_parallel':
                            pre = torch.sum(svf(params_svf_pre), -2)
                        pre = pre.transpose(-1, -2).unsqueeze(-1)
                    else:
                        params_svf_pre, pre = None, None

                    if args.post_param_per_channel != 0:
                        params_svf_post = svf_post_dec(latent)
                        if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                            post = util.prod(svf(params_svf_post).transpose(-1, -2))
                        elif args.filter == 'svf_parallel':
                            post = torch.sum(svf(params_svf_post), -2)
                        post = post.transpose(-1, -2).unsqueeze(-2)
                    else:
                        params_svf_post, post = None, None

                    if args.feedback_param_per_channel != 0:
                        if args.ds_compensation:
                            params_svf_fb = svf_fb_dec(latent, ds = params_fdn['ds'], sap_params = params_sap_fb)
                        else:
                            params_svf_fb = svf_fb_dec(latent)#, ds = params_fdn['ds'])
                        fb_filter = util.prod(svfbell(params_svf_fb).transpose(-1, -2))
                        if args.num_sap_per_channel != 0:
                            fb_filter = fb_filter * feedback_sap
                        fb_filter = fb_filter.transpose(-1, -2)
                    else:
                        params_svf_fv, fb_filter = None, None
                        if args.num_sap_per_channel != 0:
                            fb_filter = feedback_sap.transpose(-1, -2)

                params_color = dict(pre = pre, post = post, fb_filter = fb_filter)

                fdn_ir = fdn({**params_fdn, **params_color})

                loss = mss(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)), F.pad(fdn_ir, (0, args.ir_len - args.iir_sample)))
                val_losses.append(loss.detach().item())

                diff_full, diff_avg, _, p_full_rir, p_full_fdn, p_rir, p_fdn = difference_profile(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir.detach().cpu().numpy(), sr = args.sr)

                for key in diff_full.keys():
                    if not key in diff_fulls:
                        diff_fulls[key] = [diff_full[key]]
                        diff_avgs[key] = [diff_avg[key]]
                        p_full_rirs[key] = [p_full_rir[key]]
                        p_full_fdns[key] = [p_full_fdn[key]]
                        p_rirs[key] = [p_rir[key]]
                        p_fdns[key] = [p_fdn[key]]
                    else:
                        diff_fulls[key].append(diff_full[key])
                        diff_avgs[key].append(diff_avg[key])
                        p_rirs[key].append(p_rir[key])
                        p_fdns[key].append(p_fdn[key])
                        p_full_rirs[key].append(p_full_rir[key])
                        p_full_fdns[key].append(p_full_fdn[key])

                if i < 10:
                    fdn_ir = fdn_ir.detach().cpu().numpy()

                    if args.pre_param_per_channel != 0:
                        pre_filter_Bs, pre_filter_As = filt.svf(params_svf_pre['twoRs'], params_svf_pre['Gs'], params_svf_pre['c_hps'], params_svf_pre['c_bps'], params_svf_pre['c_lps'])
                    else:
                        ones = torch.ones(args.batch_size, 1, 1)
                        pre_filter_Bs, pre_filter_As = filt.svf(ones, ones, ones, ones, ones)

                    if args.post_param_per_channel != 0:
                        post_filter_Bs, post_filter_As = filt.svf(params_svf_post['twoRs'], params_svf_post['Gs'], params_svf_post['c_hps'], params_svf_post['c_bps'], params_svf_post['c_lps'])
                    else:
                        ones = torch.ones(args.batch_size, 1, 1)
                        post_filter_Bs, post_filter_As = filt.svf(ones, ones, ones, ones, ones)

                    if args.feedback_param_per_channel != 0:
                        fb_filter_Bs, fb_filter_As = filt.svf(params_svf_fb['twoRs'], params_svf_fb['Gs'], params_svf_fb['c_hps'], params_svf_fb['c_bps'], params_svf_fb['c_lps'])
                    else:
                        ones = torch.ones(args.batch_size, 1, 1)
                        fb_filter_Bs, fb_filter_As = filt.svf(ones, ones, ones, ones, ones)

                    tfdn_lti_ir, tfdn_ltv_ir = tfdn(ds = params_fdn['ds'].detach().cpu().numpy(),
                                                    pre_gain = params_fdn['pre_gain'].detach().cpu().numpy(),
                                                    pre_filter_Bs = pre_filter_Bs.detach().cpu().numpy(),
                                                    pre_filter_As = pre_filter_As.detach().cpu().numpy(),
                                                    fb_adm = params_fdn['fb_adm'].detach().cpu().numpy(),
                                                    fb_gain = params_fdn['fb_gain'].detach().cpu().numpy(),
                                                    fb_filter_Bs = fb_filter_Bs.detach().cpu().numpy(),
                                                    fb_filter_As = fb_filter_As.detach().cpu().numpy(),
                                                    #fb_SAP_alphas = params_sap_fb['alphas'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                                    #fb_SAP_ms = params_sap_fb['ds'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                                    post_gain = params_fdn['post_gain'].detach().cpu().numpy(),
                                                    post_filter_Bs = post_filter_Bs.detach().cpu().numpy(),
                                                    post_filter_As = post_filter_As.detach().cpu().numpy(),
                                                    bypass_FIR = params_fdn['fir'].detach().cpu().numpy())

                    tfdn_lti_ir, tfdn_ltv_ir = tfdn_lti_ir.detach().cpu().numpy(), tfdn_ltv_ir.detach().cpu().numpy()

                    for j in range(args.batch_size):
                        gt_ir, dspeech, tspeech = vali_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
                        cspeech = signal.convolve(dspeech, fdn_ir[j])

                        mdspeech_idx = np.random.randint(0, vali_speech_datasize)
                        _, mdspeech, _ = vali_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
                        mspeech = signal.convolve(mdspeech, fdn_ir[j])
                        tmspeech = signal.convolve(mdspeech, tfdn_lti_ir[j])
                        
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_rir.wav', gt_ir, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_transmitted_speech.wav', tspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_dry_speech.wav', dspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_estimation.wav', fdn_ir[j], args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_true_lti_estimation.wav', tfdn_lti_ir[j], args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_true_estimation.wav', tfdn_ltv_ir[j], args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_convolved_speech.wav', cspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_matching_dry_speech.wav', mdspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_matched_speech.wav', mspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_true_matched_speech.wav', mspeech, args.sr)

                    fig_edr, axes_edr = compare_edr(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir, sr = args.sr)
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
        valid_summary['p_full_rec'] = p_full_fdns
        valid_summary['p_rir'] = p_rirs
        valid_summary['p_rec'] = p_fdns
        valid_summary['rir_dir'] = vali_set.ir_directory_list

        print('epoch:', ep, "loss:", np.average(val_losses))
        print('DIFF')

        for key in diff_avgs.keys():
            print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
            print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))

        ckpt = dict(enc = enc.state_dict(),
                    fdn_dec = fdn_dec.state_dict(),
                    svf_pre_dec = svf_pre_dec.state_dict() if args.pre_param_per_channel != 0 else None,
                    svf_post_dec = svf_post_dec.state_dict() if args.post_param_per_channel != 0 else None,
                    svf_fb_dec = svf_fb_dec.state_dict() if args.feedback_param_per_channel != 0 else None,
                    sap_dec = sap_dec.state_dict() if args.num_sap_per_channel != 0 else None,
                    opt = optim.state_dict(),
                    scd = sched.state_dict())

        #if args.pre_param_per_channel != 0:
        #    ckpt['svf_pre_dec'] = svf_pre_dec.state_dict()
        #if args.post_param_per_channel != 0:
        #    ckpt['svf_post_dec'] = svf_post_dec.state_dict()
        #if args.feedback_param_per_channel != 0:
        #    ckpt['svf_fb_dec'] = svf_fb_dec.state_dict()
        #if args.num_sap_per_channel != 0:
        #    ckpt['sap_dec'] = sap_dec.state_dict()

        torch.save(ckpt, save_dir + 'checkpoints/' + str(ep).zfill(2) + '.pt')

        with open(vali_dir + 'result.pickle', 'wb') as fp:
            pickle.dump(valid_summary, fp)

""" TEST """ 
test_losses, true_lti_losses, true_ltv_losses = [], [], []

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
        D = self.logspec(ir)**2
        D = torch.flip(torch.cumsum(torch.flip(D, (-1,)), -1), (-1,))
        return D 

logedr = LogEDRDifference().cuda()
logedr_differences = []

if args.blind_estimation:
    task_str = 'be_'
else:
    task_str = 'as_'
with torch.no_grad():
    diff_fulls, diff_avgs, p_full_rirs, p_full_fdns, p_rirs, p_fdns = {}, {}, {}, {}, {}, {}
    t_lti_diff_fulls, t_lti_diff_avgs, p_full_t_lti_fdns, p_t_lti_fdns = {}, {}, {}, {}
    t_ltv_diff_fulls, t_ltv_diff_avgs, p_full_t_ltv_fdns, p_t_ltv_fdns = {}, {}, {}, {}

    for i, (ir, tspeech, dspeech, dspeech_idx, ) in enumerate(tqdm(test_loader)):
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
                latent = enc(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)))
        else:
            tspeech = tspeech.float()
            ir = ir.float()
            random = (torch.rand(4, 1) > 0.5).float()
            mixed = ir * random + tspeech * (1 - random) 
            ir = ir.cuda()
            mixed = mixed.cuda()
            latent = enc(mixed)


        params_fdn = fdn_dec(latent)

        if args.num_sap_per_channel != 0:
            params_sap_fb = sap_dec(latent)
            feedback_sap = util.prod(sap(params_sap_fb).transpose(-1, -2))
        else:
            params_sap_fb = None

        if args.filter == 'svf_parallel' or args.filter == 'svf_serial' or args.filter == 'parametric_eq':
            if args.pre_param_per_channel != 0:
                params_svf_pre = svf_pre_dec(latent)
                if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                    pre = util.prod(svf(params_svf_pre).transpose(-1, -2))
                elif args.filter == 'svf_parallel':
                    pre = torch.sum(svf(params_svf_pre), -2)
                pre = pre.transpose(-1, -2).unsqueeze(-1)
            else:
                params_svf_pre, pre = None, None

            if args.post_param_per_channel != 0:
                params_svf_post = svf_post_dec(latent)
                if args.filter == 'svf_serial' or args.filter == 'parametric_eq':
                    post = util.prod(svf(params_svf_post).transpose(-1, -2))
                elif args.filter == 'svf_parallel':
                    post = torch.sum(svf(params_svf_post), -2)
                post = post.transpose(-1, -2).unsqueeze(-2)
            else:
                params_svf_post, post = None, None

            if args.feedback_param_per_channel != 0:
                if args.ds_compensation:
                    print('fb')
                    params_svf_fb = svf_fb_dec(latent, ds = params_fdn['ds'], sap_params = params_sap_fb)
                    print('fb')
                else:
                    params_svf_fb = svf_fb_dec(latent)#, ds = params_fdn['ds'])
                fb_filter = util.prod(svfbell(params_svf_fb).transpose(-1, -2))
                if args.num_sap_per_channel != 0:
                    fb_filter = fb_filter * feedback_sap
                fb_filter = fb_filter.transpose(-1, -2)
            else:
                params_svf_fb, fb_filter = None, None
                if args.num_sap_per_channel != 0:
                    fb_filter = feedback_sap.transpose(-1, -2)

        params_color = dict(pre = pre, post = post, fb_filter = fb_filter)

        fdn_ir = fdn({**params_fdn, **params_color})
        fdn_ir = fdn_ir[:, :args.ir_len]
        """
        true fdn
        """

        if args.pre_param_per_channel != 0:
            pre_filter_Bs, pre_filter_As = filt.svf(params_svf_pre['twoRs'], params_svf_pre['Gs'], params_svf_pre['c_hps'], params_svf_pre['c_bps'], params_svf_pre['c_lps'])
        else:
            ones = torch.ones(args.batch_size, 1, 1)
            pre_filter_Bs, pre_filter_As = filt.svf(ones, ones, ones, ones, ones)

        if args.post_param_per_channel != 0:
            post_filter_Bs, post_filter_As = filt.svf(params_svf_post['twoRs'], params_svf_post['Gs'], params_svf_post['c_hps'], params_svf_post['c_bps'], params_svf_post['c_lps'])
        else:
            ones = torch.ones(args.batch_size, 1, 1)
            post_filter_Bs, post_filter_As = filt.svf(ones, ones, ones, ones, ones)

        if args.feedback_param_per_channel != 0:
            fb_filter_Bs, fb_filter_As = filt.svf(params_svf_fb['twoRs'], params_svf_fb['Gs'], params_svf_fb['c_hps'], params_svf_fb['c_bps'], params_svf_fb['c_lps'])
            print('x', fb_filter_Bs[0][0], fb_filter_As[0][0])
        else:
            ones = torch.ones(args.batch_size, 1, 1)
            fb_filter_Bs, fb_filter_As = filt.svf(ones, ones, ones, ones, ones)

        if not args.mturk:
            tfdn_lti_ir, tfdn_ltv_ir = tfdn(ds = params_fdn['ds'].detach().cpu().numpy(),
                                            pre_gain = params_fdn['pre_gain'].detach().cpu().numpy(),
                                            pre_filter_Bs = pre_filter_Bs.detach().cpu().numpy(),
                                            pre_filter_As = pre_filter_As.detach().cpu().numpy(),
                                            fb_adm = params_fdn['fb_adm'].detach().cpu().numpy(),
                                            fb_gain = params_fdn['fb_gain'].detach().cpu().numpy(),
                                            fb_filter_Bs = fb_filter_Bs.detach().cpu().numpy(),
                                            fb_filter_As = fb_filter_As.detach().cpu().numpy(),
                                            fb_SAP_alphas = params_sap_fb['alphas'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                            fb_SAP_ms = params_sap_fb['ds'].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                            post_gain = params_fdn['post_gain'].detach().cpu().numpy(),
                                            post_filter_Bs = post_filter_Bs.detach().cpu().numpy(),
                                            post_filter_As = post_filter_As.detach().cpu().numpy(),
                                            bypass_FIR = params_fdn['fir'].detach().cpu().numpy())
        
            loss = mss(F.pad(ir[:, :args.iir_sample], (0, args.ir_len - args.iir_sample)), F.pad(fdn_ir, (0, args.ir_len - args.iir_sample)))
            test_losses.append(loss.detach().item())


            true_lti_loss = mss(ir, tfdn_lti_ir.float()).detach().item()
            true_lti_losses.append(true_lti_loss)

            true_ltv_loss = mss(ir, tfdn_ltv_ir.float()).detach().item()
            true_ltv_losses.append(true_ltv_loss)

            logedr_differences.append(logedr(ir, tfdn_ltv_ir.float()).item())

            print(loss.detach().item(), true_lti_loss, true_ltv_loss)

            figtemp, axtemp = plt.subplots(1, 3,sharey = True)
            axtemp[0].plot(tfdn_lti_ir.detach().cpu().numpy()[0][:2000])
            axtemp[1].plot(fdn_ir.detach().cpu().numpy()[0][:2000])
            axtemp[2].plot(fdn_ir.detach().cpu().numpy()[0][:2000] - tfdn_lti_ir.detach().cpu().numpy()[0][:2000])
            figtemp.savefig('wow.pdf')

            diff_full, diff_avg, _, p_full_rir, p_full_fdn, p_rir, p_fdn = difference_profile(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir.detach().cpu().numpy(), sr = args.sr)
            t_lti_diff_full, t_lti_diff_avg, _, _, p_full_t_lti_fdn, _, p_t_lti_fdn = difference_profile(ir.detach().cpu().numpy(), tfdn_lti_ir.detach().cpu().numpy(), sr = args.sr) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            t_ltv_diff_full, t_ltv_diff_avg, _, _, p_full_t_ltv_fdn, _, p_t_ltv_fdn = difference_profile(ir.detach().cpu().numpy(), tfdn_ltv_ir.detach().cpu().numpy(), sr = args.sr) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for key in diff_full.keys():
                if not key in diff_fulls:
                    diff_fulls[key] = [diff_full[key]]
                    diff_avgs[key] = [diff_avg[key]]
                    p_full_rirs[key] = [p_full_rir[key]]
                    p_full_fdns[key] = [p_full_fdn[key]]
                    p_rirs[key] = [p_rir[key]]
                    p_fdns[key] = [p_fdn[key]]

                    t_lti_diff_fulls[key] = [t_lti_diff_full[key]]
                    t_lti_diff_avgs[key] = [t_lti_diff_avg[key]]
                    p_full_t_lti_fdns[key] = [p_full_t_lti_fdn[key]]
                    p_t_lti_fdns[key] = [p_t_lti_fdn[key]]

                    t_ltv_diff_fulls[key] = [t_ltv_diff_full[key]]
                    t_ltv_diff_avgs[key] = [t_ltv_diff_avg[key]]
                    p_full_t_ltv_fdns[key] = [p_full_t_ltv_fdn[key]]
                    p_t_ltv_fdns[key] = [p_t_ltv_fdn[key]]

                else:
                    diff_fulls[key].append(diff_full[key])
                    diff_avgs[key].append(diff_avg[key])
                    p_rirs[key].append(p_rir[key])
                    p_fdns[key].append(p_fdn[key])
                    p_full_rirs[key].append(p_full_rir[key])
                    p_full_fdns[key].append(p_full_fdn[key])

                    t_lti_diff_fulls[key].append(t_lti_diff_full[key])
                    t_lti_diff_avgs[key].append(t_lti_diff_avg[key])
                    p_full_t_lti_fdns[key].append(p_full_t_lti_fdn[key])
                    p_t_lti_fdns[key].append(p_t_lti_fdn[key])

                    t_ltv_diff_fulls[key].append(t_ltv_diff_full[key])
                    t_ltv_diff_avgs[key].append(t_ltv_diff_avg[key])
                    p_full_t_ltv_fdns[key].append(p_full_t_ltv_fdn[key])
                    p_t_ltv_fdns[key].append(p_t_ltv_fdn[key])

            fdn_ir = fdn_ir.detach().cpu().numpy()
            tfdn_lti_ir = tfdn_lti_ir.detach().cpu().numpy() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            tfdn_ltv_ir = tfdn_ltv_ir.detach().cpu().numpy() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for j in range(args.batch_size):
            gt_ir, dspeech, tspeech = test_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
            if not args.mturk:
                cspeech = signal.convolve(dspeech, fdn_ir[j])

            tcspeech = tfdn.render(ds = params_fdn['ds'][j].detach().cpu().numpy(),
                                   pre_gain = params_fdn['pre_gain'][j].detach().cpu().numpy(),
                                   pre_filter_Bs = pre_filter_Bs[j].detach().cpu().numpy(),
                                   pre_filter_As = pre_filter_As[j].detach().cpu().numpy(),
                                   fb_adm = params_fdn['fb_adm'][j].detach().cpu().numpy(),
                                   fb_gain = params_fdn['fb_gain'][j].detach().cpu().numpy(),
                                   fb_filter_Bs = fb_filter_Bs[j].detach().cpu().numpy(),
                                   fb_filter_As = fb_filter_As[j].detach().cpu().numpy(),
                                   fb_SAP_alphas = params_sap_fb['alphas'][j].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                   fb_SAP_ms = params_sap_fb['ds'][j].detach().cpu().numpy() if args.num_sap_per_channel != 0 else None,
                                   post_gain = params_fdn['post_gain'][j].detach().cpu().numpy(),
                                   post_filter_Bs = post_filter_Bs[j].detach().cpu().numpy(),
                                   post_filter_As = post_filter_As[j].detach().cpu().numpy(),
                                   bypass_FIR = params_fdn['fir'][j].detach().cpu().numpy(),
                                   input_audio = dspeech)
            tcspeech = tcspeech.detach().cpu().numpy().squeeze()

            #mdspeech_idx = np.random.randint(0, test_speech_datasize)
            #_, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i)
            #mspeech = signal.convolve(mdspeech, fdn_ir[j])
            #tmspeech = signal.convolve(mdspeech, tfdn_lti_ir[j])

            #_, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
            if not args.mturk:
                sf.write(test_dir + task_str + str(j) + '_rir.wav', gt_ir, args.sr)
                sf.write(test_dir + task_str + str(j) + '_rir_crop.wav', ir[j][:args.iir_sample].detach().cpu().numpy(), args.sr)
                sf.write(test_dir + task_str + str(j) + '_transmitted_speech.wav', rms_normalize(tspeech), args.sr)
                sf.write(test_dir + task_str + str(j) + '_dry_speech.wav', rms_normalize(dspeech), args.sr)
                sf.write(test_dir + task_str + str(j) + '_freq_sampled_estimation.wav', fdn_ir[j], args.sr)
                sf.write(test_dir + task_str + str(j) + '_freq_sampled_convolved_speech.wav', rms_normalize(cspeech), args.sr)
                sf.write(test_dir + task_str + str(j) + '_true_lti_estimation.wav', tfdn_lti_ir[j], args.sr)
                sf.write(test_dir + task_str + str(j) + '_true_estimation.wav', tfdn_ltv_ir[j], args.sr)
            #sf.write(test_dir + task_str + str(j) + '_matching_dry_speech.wav', mdspeech, args.sr)
            #sf.write(test_dir + task_str + str(j) + '_freq_sampled_matched_speech.wav', mspeech, args.sr)
            sf.write(test_dir + task_str + str(j) + '_true_convolved_speech.wav', rms_normalize(tcspeech), args.sr)
                #sf.write(test_dir + task_str + str(j) + '_true_matched_speech.wav', tmspeech, args.sr)

        if not args.mturk:
            fig_edr, axes_edr = compare_edr(ir[:, :args.iir_sample].detach().cpu().numpy(), fdn_ir, sr = args.sr)
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(test_dir + 'edr' + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)

            fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), tfdn_lti_ir, sr = args.sr)
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(test_dir + 'tltiedr' + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)

            fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), tfdn_ltv_ir, sr = args.sr)
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(test_dir + 'tltvedr' + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)

            if args.pre_param_per_channel != 0:
                for key in params_fdn.keys():
                    params_fdn[key] = params_fdn[key].clone().detach().cpu().numpy()

            if args.post_param_per_channel != 0:
                for key in params_svf_post.keys():
                    params_svf_post[key] = params_svf_post[key].clone().detach().cpu().numpy()

            if args.feedback_param_per_channel != 0:
                for key in params_svf_fb.keys():
                    params_svf_fb[key] = params_svf_fb[key].clone().detach().cpu().numpy()

            total_params = dict(fdn = params_fdn, post = params_svf_post, fb = params_svf_fb)
            pickle.dump(total_params, open(test_dir + 'param.pickle', 'wb'))

        if args.debug:
            break
            
test_summary = {}
test_summary['loss'] = np.average(test_losses)
test_summary['true_lti_loss'] = np.average(true_lti_losses)
test_summary['true_ltv_loss'] = np.average(true_ltv_losses)
test_summary['diff_full'] = diff_fulls
test_summary['diff_avg'] = diff_avgs
test_summary['true_lti_diff_full'] = t_lti_diff_fulls
test_summary['true_lti_diff_avg'] = t_lti_diff_avgs
test_summary['true_ltv_diff_full'] = t_ltv_diff_fulls
test_summary['true_ltv_diff_avg'] = t_ltv_diff_avgs
test_summary['p_full_rir'] = p_full_rirs
test_summary['p_full_rec'] = p_full_fdns
test_summary['p_full_true_lti_rec'] = p_full_t_lti_fdns
test_summary['p_full_true_ltv_rec'] = p_full_t_ltv_fdns
test_summary['p_rir'] = p_rirs
test_summary['p_rec'] = p_fdns
test_summary['p_true_lti_rec'] = p_t_lti_fdns
test_summary['p_true_ltv_rec'] = p_t_ltv_fdns
test_summary['rir_dir'] = test_set.ir_directory_list

print('final loss:', np.average(test_losses))
print('final true lti loss:', np.average(true_lti_losses))
print('final true ltv loss:', np.average(true_ltv_losses))
print('DIFF')

for key in diff_avgs.keys():
    #print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
    print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))
for key in t_lti_diff_avgs.keys():
    #print(key, '(true mean) %.3f' % np.average(np.concatenate(t_lti_diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(t_lti_diff_avgs[key], 0)))
    print(key, '(true lti median) %.3f' % np.median(np.concatenate(t_lti_diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(t_lti_diff_avgs[key], 0)))
    print(key, '(true ltv median) %.3f' % np.median(np.concatenate(t_ltv_diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(t_ltv_diff_avgs[key], 0)))
print(np.average(logedr_differences))

with open(save_dir + 'test/result.pickle', 'wb') as fp:
    pickle.dump(test_summary, fp)
