import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class FrameAttentionModule(nn.Module):
    def __init__(self,in_channels, mode = 'relation'):
        """
        https://arxiv.org/pdf/1907.00193.pdf
        Reimplemented by SeongwoongJo
        
        """
        super(FrameAttentionModule,self).__init__()
        self.attn_mode = mode
        self.l1 = nn.Linear(in_channels, 1)
        if mode == 'relation':
            self.l2 = nn.Linear(2*in_channels, 1)
    def forward(self,fs):
        ## fs : B,T,F
        alphas = F.sigmoid(self.l1(fs)) + 1e-3 # B,T,1
        scores = alphas/torch.sum(alphas,dim=1,keepdim=True)
#        scores = F.softmax(alphas, dim = 1) # B,T,1
        fv_hat = torch.sum(scores * fs, dim = 1) # B,F
        
        if self.attn_mode == 'self':
            return fv_hat
        
        T = fs.shape[1]
        fv = torch.cat([fv_hat.unsqueeze(1)]*T, dim = 1) # B,T,F
        fv = torch.cat([fv,fs],dim = -1) # B,T,2F
        betas = F.sigmoid(self.l2(fv)) + 1e-3 # B,T,1
        
#        scores = F.softmax(alphas * betas, dim = 1) # B,T,1
        scores = alphas*betas / torch.sum(alphas*betas, dim = 1,keepdim=True)
        fv = torch.sum(scores*fv, dim = 1) # B,2F
        
        return fv
    
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ResBlock(dim, norm=norm, act=act, pad_type=pad_type, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act=act,
                                               pad_type=pad_type, use_sn=use_sn),
                                   Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act='none',
                                               pad_type=pad_type, use_sn=use_sn))

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out



class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ContentEncoder(nn.Module):
    def __init__(self, nf_cnt, n_downs, n_res, norm, act, pad, use_sn=False):
        super(ContentEncoder, self).__init__()
        print("Init ContentEncoder")

        nf = nf_cnt

        self.model = nn.ModuleList()
        self.model.append(Conv2dBlock(3, nf, 7, 1, 3, norm=norm, act=act, pad_type=pad, use_sn=use_sn))

        for _ in range(n_downs):
            self.model.append(Conv2dBlock(nf, 2 * nf, 4, 2, 1, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
            nf *= 2
        
        self.model.append(ResBlocks(n_res, nf, norm=norm, act=act, pad_type=pad, use_sn=use_sn))

        self.model = nn.Sequential(*self.model)
        self.out_dim = nf

    def forward(self, x):
        x = self.model(x)
        return torch.mean(x,dim = (-1,-2))

