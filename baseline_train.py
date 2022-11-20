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

from train_reverb.network.baseline import TimeDomainBaseline3 as TimeDomainBaseline
from train_reverb.network.encoder_rnn_optimized_2 import Encoder
from train_reverb.loss.mss_loss import MSSLoss5 as MSSLoss ## refactor !
from train_reverb.loss.mss_loss import DecayCompensator

from train_reverb.metric.metric_new_2 import difference_profile 
from train_reverb.metric.plot_edr import compare_edr
from train_reverb.metric.plot_training_status import *
import components.utils as util
from utils import rms_normalize

from tqdm import tqdm
import pickle
from scipy import signal

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

parser.add_argument('--encoder_spec', type = str, default = 'log', help = 'frequency-axis scale of encoder spectrogram - lin, mel, log or cqt')

parser.add_argument('--weighted_loss', type = bool, default = False, help = 'weighted loss for decay compensation')
parser.add_argument('--loss_spec', type = str, default = 'log', help = 'frequency-axis scale of spectrogram loss - lin, mel, log or cqt')
parser.add_argument('--loss_nfft', type = int, nargs = '+', default = [256, 512, 1024, 2048, 4096], help = 'nfft of MSSLoss') # more detailed? 8192?
parser.add_argument('--loss_alpha', type = float, default = 0., help = 'alpha (weight of log loss term) of MSSLoss')
parser.add_argument('--lr', type = float, default = 1e-5, help = 'initial lr of optimizer')


parser.add_argument('--print_loss_per', type = int, default = 1000, help = 'print out average of recent losses')
parser.add_argument('--plot_status_per', type = int, default = 1000, help = 'plot and save training status')
parser.add_argument('--checkpoint', type = str, default = None, help = 'checkpoint of saved model')
parser.add_argument('--num_sample_train_jndr', type = int, default = 200, help = 'num of samples for calculating jndr at train set')

args = parser.parse_args()

if args.mturk:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat_mturk import Generate_Dataset
else:
    from train_reverb.dataset.speech_to_ir.pyroom_3_4_saved_cat import Generate_Dataset

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
dec = TimeDomainBaseline().cuda()

if args.blind_estimation:
    input_shape = (109, 256)
else:
    input_shape = (109, 256)

net_params = list(enc.parameters()) + list(dec.parameters())
total_num_params = sum(p.numel() for p in net_params if p.requires_grad)

comp = DecayCompensator().cuda()
mss = MSSLoss(args.loss_nfft, args.loss_spec, alpha = args.loss_alpha, sr = args.sr).cuda()
optim = torch.optim.Adam(params = net_params, lr = args.lr)
if args.blind_estimation:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], gamma = 0.1 ** (0.1), verbose = True)
else:
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [5, 6, 7, 8, 9], gamma = 0.1 ** (0.2), verbose = True)

print("TOTAL PARAMS", total_num_params)

if cont:
    enc.load_state_dict(ckpt['enc'], strict=False)
    dec.load_state_dict(ckpt['dec'])
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
        optim.zero_grad()
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

        ir_recon = dec(latent)
        match_loss = mss(ir, ir_recon)

        if i % args.plot_status_per == 0:
            print("match", match_loss.detach().item())

        match_losses.append(match_loss.detach().item())
        loss = match_loss

        loss.backward()
        optim.step()

        if i < args.num_sample_train_jndr:
            diff_full, diff_avg, _, _, _, _, _ = difference_profile(ir.detach().cpu().numpy(), ir_recon.detach().cpu().numpy(), sr = args.sr)

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
            print(nowstr, ep, i, np.average(match_losses))
            match_losses  = []

        if i % args.plot_status_per == 0: #and (ep != 0 or (ep == 0 and i != 0)):
            fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), ir_recon.detach().cpu().numpy())
            fig_edr.set_size_inches(20, 12)
            fig_edr.savefig(train_dir + 'edr_' + str(i).zfill(5) + '.pdf', bbox_inches = 'tight')
            plt.close(fig_edr)

        if i % args.plot_status_per == 0:
            for j in range(args.batch_size):
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_rir.wav', ir[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_freq_sampled_estimation.wav', ir_recon[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_transmitted_speech.wav', tspeech[j].detach().cpu().numpy(), args.sr)
                sf.write(train_dir + str(i).zfill(5) + '_' + str(j) + '_dry_speech.wav', dspeech[j].detach().cpu().float().numpy(), args.sr)

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
            diff_fulls, diff_avgs, p_full_rirs, p_full_recons, p_rirs, p_recons = {}, {}, {}, {}, {}, {}
            for i, (ir, tspeech, dspeech, dspeech_idx) in enumerate(tqdm(vali_loader)):
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

                ir_recon = dec(latent)

                loss = mss(ir, ir_recon)
                val_losses.append(loss.detach().item())

                diff_full, diff_avg, _, p_full_rir, p_full_recon, p_rir, p_recon = difference_profile(ir.detach().cpu().numpy(), ir_recon.detach().cpu().numpy(), sr = args.sr)
                ir_recon = ir_recon.detach().cpu().numpy()

                for key in diff_full.keys():
                    if not key in diff_fulls:
                        diff_fulls[key] = [diff_full[key]]
                        diff_avgs[key] = [diff_avg[key]]
                        p_full_rirs[key] = [p_full_rir[key]]
                        p_full_recons[key] = [p_full_recon[key]]
                        p_rirs[key] = [p_rir[key]]
                        p_recons[key] = [p_recon[key]]
                    else:
                        diff_fulls[key].append(diff_full[key])
                        diff_avgs[key].append(diff_avg[key])
                        p_rirs[key].append(p_rir[key])
                        p_recons[key].append(p_recon[key])
                        p_full_rirs[key].append(p_full_rir[key])
                        p_full_recons[key].append(p_full_recon[key])

                if i < 10:
                    for j in range(args.batch_size):
                        gt_ir, dspeech, tspeech = vali_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
                        cspeech = signal.convolve(dspeech, ir_recon[j])

                        mdspeech_idx = np.random.randint(0, vali_speech_datasize)
                        _, mdspeech, _ = vali_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
                        mspeech = signal.convolve(mdspeech, ir_recon[j])
                        
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_rir.wav', gt_ir, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_transmitted_speech.wav', tspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_dry_speech.wav', dspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_estimation.wav', ir_recon[j], args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_freq_sampled_convolved_speech.wav', cspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_matching_dry_speech.wav', mdspeech, args.sr)
                        sf.write(vali_dir + str(i) + '_' + str(j) + '_true_matched_speech.wav', mspeech, args.sr)

                    fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), ir_recon, sr = args.sr)
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
        valid_summary['p_full_rec'] = p_full_recons
        valid_summary['p_rir'] = p_rirs
        valid_summary['p_rec'] = p_recons
        valid_summary['rir_dir'] = vali_set.ir_directory_list

        print('epoch:', ep, "loss:", np.average(val_losses))
        print('DIFF')

        for key in diff_avgs.keys():
            print(key, '(mean) %.3f' % np.average(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.average(np.concatenate(diff_avgs[key], 0)))
            print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))

        ckpt = dict(enc = enc.state_dict(),
                    dec = dec.state_dict())

        torch.save(ckpt, save_dir + 'checkpoints/' + str(ep).zfill(2) + '.pt')

        with open(vali_dir + 'result.pickle', 'wb') as fp:
            pickle.dump(valid_summary, fp)

""" TEST """ 
test_losses, true_lti_losses, true_ltv_losses = [], [], []

if args.blind_estimation:
    task_str = 'be_'
else:
    task_str = 'as_'
with torch.no_grad():
    diff_fulls, diff_avgs, p_full_rirs, p_full_recons, p_rirs, p_recons = {}, {}, {}, {}, {}, {}

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
                latent = enc(ir)
        else:
            tspeech = tspeech.float()
            ir = ir.float()
            random = (torch.rand(4, 1) > 0.5).float()
            mixed = ir * random + tspeech * (1 - random) 
            ir = ir.cuda()
            mixed = mixed.cuda()
            latent = enc(mixed)

        ir_recon = dec(latent)

        ir_recon = ir_recon.detach().cpu().numpy()
        lpf_sos = signal.butter(6, 22000 / 48000, 'lowpass', output = 'sos')
        hpf_sos = signal.butter(6, 40 / 48000, 'highpass', output = 'sos')
        ir_recon = signal.sosfilt(lpf_sos, ir_recon)[:, ::-1]
        ir_recon = signal.sosfilt(lpf_sos, ir_recon)[:, ::-1]
        ir_recon = signal.sosfilt(hpf_sos, ir_recon)[:, ::-1]
        ir_recon = signal.sosfilt(hpf_sos, ir_recon)[:, ::-1]
        sf.write('wtf.wav', ir_recon[0], 48000)
        loss = mss(ir, torch.tensor(ir_recon.copy()).float().cuda())
        test_losses.append(loss.item())

        print(loss.item())

        diff_full, diff_avg, _, p_full_rir, p_full_recon, p_rir, p_recon = difference_profile(ir.detach().cpu().numpy(), ir_recon, sr = args.sr)
        #print(diff_full, diff_avg)
        #print(test_dir)

        for key in diff_full.keys():
            if not key in diff_fulls:
                diff_fulls[key] = [diff_full[key]]
                diff_avgs[key] = [diff_avg[key]]
                p_full_rirs[key] = [p_full_rir[key]]
                p_full_recons[key] = [p_full_recon[key]]
                p_rirs[key] = [p_rir[key]]
                p_recons[key] = [p_recon[key]]
            else:
                diff_fulls[key].append(diff_full[key])
                diff_avgs[key].append(diff_avg[key])
                p_rirs[key].append(p_rir[key])
                p_recons[key].append(p_recon[key])
                p_full_rirs[key].append(p_full_rir[key])
                p_full_recons[key].append(p_full_recon[key])

        for j in range(args.batch_size):
            gt_ir, dspeech, tspeech = test_set.get_full_data(dspeech_idx[j], i * args.batch_size + j)
            cspeech = signal.convolve(dspeech, ir_recon[j])

            mdspeech_idx = np.random.randint(0, test_speech_datasize)
            _, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i)
            mspeech = signal.convolve(mdspeech, ir_recon[j])

            _, mdspeech, _ = test_set.get_full_data(mdspeech_idx, i * args.batch_size + j)
            sf.write(test_dir + task_str + str(j) + '_rir.wav', gt_ir, args.sr)
            sf.write(test_dir + task_str + str(j) + '_rir_crop.wav', ir[j].detach().cpu().numpy(), args.sr)
            sf.write(test_dir + task_str + str(j) + '_transmitted_speech.wav', rms_normalize(tspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_dry_speech.wav', rms_normalize(dspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_matching_dry_speech.wav', rms_normalize(mdspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_estimation.wav', ir_recon[j], args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_convolved_speech.wav', rms_normalize(cspeech), args.sr)
            sf.write(test_dir + task_str + str(j) + '_freq_sampled_matched_speech.wav', rms_normalize(mspeech), args.sr)
            print(test_dir + task_str + str(j))

        fig_edr, axes_edr = compare_edr(ir.detach().cpu().numpy(), ir_recon, sr = args.sr)
        fig_edr.set_size_inches(20, 12)
        fig_edr.savefig(test_dir + 'edr' + '.pdf', bbox_inches = 'tight')
        plt.close(fig_edr)

        if args.debug:
            break
            
test_summary = {}
test_summary['loss'] = np.average(test_losses)
test_summary['diff_full'] = diff_fulls
test_summary['diff_avg'] = diff_avgs
test_summary['p_full_rir'] = p_full_rirs
test_summary['p_full_rec'] = p_full_recons
test_summary['p_rir'] = p_rirs
test_summary['p_rec'] = p_recons
test_summary['rir_dir'] = test_set.ir_directory_list

print('final loss:', np.average(test_losses))
print('DIFF')

for key in diff_avgs.keys():
    print(key, '(median) %.3f' % np.median(np.concatenate(diff_fulls[key], 0)), '%.3f' % np.median(np.concatenate(diff_avgs[key], 0)))
with open(save_dir + 'test/result.pickle', 'wb') as fp:
    pickle.dump(test_summary, fp)
