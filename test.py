import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
from fvn import FVN, FVN_Decoder
from fdn import FDN, FDN_Decoder
from svf import SVF, SVF_Decoder, PEQ_Decoder
from sap import SAP, SAP_Decoder
import util
from encoder import Encoder

device = 'cpu'
input_shape = (109, 256)

def test_fvn():
    encoder = Encoder(sr=48000, n_fft=1024)
    fvn_dec = FVN_Decoder(input_shape=input_shape, num_seg=20, gain_per_seg=4, fir_len=50).to(device)
    fvn = FVN(seg_num=20, gain_per_seg=4, ir_length=120000, nonuniform_segmentation=True,
              num_sap=20, cumulative_sap=True, device=device).to(device)
    svf_dec = SVF_Decoder(input_shape=input_shape, num_channel=20, svf_per_channel=8, parallel=False,
                          sr=48000, f_init=(40, 12000)).to(device)
    svf = SVF(num_sample=4000).to(device)

    reference = torch.rand(4, 120000)
    latent = encoder(reference)
    params_fvn = fvn_dec(latent)
    params_svf = svf_dec(latent)
    color = svf(params_svf)
    color = util.prod(color.transpose(-1, -2))
    fvn_ir = fvn(params_fvn, color) 
    fvn_ir = fvn_ir[:, :120000]
    return fvn_ir

def test_advfvn():
    encoder = Encoder(sr=48000, n_fft=1024)
    init_num_svf, delta_num_svf = 8, 2
    fvn_dec = FVN_Decoder(input_shape=input_shape, num_seg=20, gain_per_seg=4, fir_len=50).to(device)
    fvn = FVN(seg_num=20, gain_per_seg=4, ir_length=120000, nonuniform_segmentation=True, num_sap=20, 
              cumulative_sap=True, device=device).to(device)
    svf_dec = SVF_Decoder(input_shape=input_shape, num_channel=1,
                          svf_per_channel=init_num_svf+delta_num_svf*19, 
                          parallel=False,
                          sr=48000,
                          f_init=(40, 12000)).to(device)
    svf = SVF(num_sample=4000).to(device)

    reference = torch.rand(4, 120000)
    latent = encoder(reference)
    params_fvn = fvn_dec(latent)
    params_svf = svf_dec(latent)
    color = svf(params_svf)
    color = util.cumprod(color.transpose(-1, -2)).squeeze(-3)
    color = color.transpose(-1, -2)[:, init_num_svf - 1::delta_num_svf, :]
    advfvn_ir = fvn(params_fvn, color) 
    advfvn_ir = advfvn_ir[:, :120000]
    return advfvn_ir

def test_fdn():
    encoder = Encoder(sr=48000, n_fft=1024)
    fdn_dec = FDN_Decoder(input_shape=input_shape, num_channel=6,
                          fir_len=100, trainable_admittance=False).to(device)
    fdn = FDN(num_channel=6, num_sample=120000).to(device)
    svf_fb_dec = PEQ_Decoder(input_shape=input_shape, 
                             num_channel=1,
                             svf_per_channel=8,
                             sr=48000,
                             f_init=(40, 12000)).to(device)
    svf_post_dec = SVF_Decoder(input_shape=input_shape, 
                               num_channel=1,
                               parallel=False,
                               sr=48000,
                               f_init=(40, 12000),
                               svf_per_channel=8,
                               advanced=False).to(device)
    svf = SVF(num_sample=120000).to(device)
    sap_dec = SAP_Decoder(input_shape, num_channel=6, sap_per_channel=4).to(device)
    sap = SAP(num_sample=120000).to(device)

    reference = torch.rand(4, 120000)
    latent = encoder(reference)
    params_fdn = fdn_dec(latent)
    params_sap_fb = sap_dec(latent)
    feedback_sap = util.prod(sap(params_sap_fb).transpose(-1, -2))
    params_svf_pre, pre = None, None
    params_svf_post = svf_post_dec(latent)
    post = util.prod(svf(params_svf_post).transpose(-1, -2))
    post = post.transpose(-1, -2).unsqueeze(-2)
    params_svf_fb = svf_fb_dec(latent)
    fb_filter = util.prod(svf(params_svf_fb).transpose(-1, -2))
    fb_filter = fb_filter * feedback_sap
    fb_filter = fb_filter.transpose(-1, -2)
    params_color = dict(pre=pre, post=post, fb_filter=fb_filter)
    fdn_ir = fdn({**params_fdn, **params_color})
    fdn_ir = fdn_ir[:, :120000]
    return fdn_ir

if __name__ == '__main__':
    fvn_ir = test_fvn().squeeze().detach().numpy()
    advfvn_ir = test_advfvn().squeeze().detach().numpy()
    fdn_ir = test_fdn().squeeze().detach().numpy()
    print(fvn_ir.shape, advfvn_ir.shape, fdn_ir.shape)
