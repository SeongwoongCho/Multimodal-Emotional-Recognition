import torch
import torch.nn as nn
import torch.nn.functional as F

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
        alphas = F.sigmoid(self.l1(fs)) # B,T,1
        scores = F.softmax(alphas, dim = 1) # B,T,1
        fv_hat = torch.sum(scores * fs, dim = 1) # B,F
        
        if self.attn_mode == 'self':
            return fv_hat
        
        T = fs.shape[1]
        fv = torch.cat([fv_hat.unsqueeze(1)]*T, dim = 1) # B,T,F
        fv = torch.cat([fv,fs],dim = -1) # B,T,2F
        betas = F.sigmoid(self.l2(fv)) # B,T,1
        
        scores = F.softmax(alphas * betas, dim = 1) # B,T,1
        fv = torch.sum(scores*fv, dim = 1) # B,2F
        
        return fv