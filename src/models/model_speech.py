from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from .attention import attention_bilstm
import torch.nn as nn
import torch

n_classes = 7

def get_speech_model(coeff = 0, weights = None):
    model = attentioned_CNN(coeff = coeff)
    if weights:
        model.load_state_dict(torch.load(weights)['model_state_dict'])
    
    return model

class attentioned_CNN(nn.Module):
    def __init__(self,coeff = 0):
        super(attentioned_CNN,self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.model._conv_stem = Conv2dStaticSamePadding(1,32, kernel_size = (3,3), stride = (2,2), bias = False,image_size = 128)
        self.out = attention_bilstm(num_classes = 7,input_size = self.model._fc.in_features, hidden_size = 512)
        del self.model._fc
    def forward(self,x,extract_feature = False):
        x = self.model.extract_features(x) # B,C,F,T
        x = torch.mean(x,dim = 2) # B,C,T 
        x = self.out(x,extract_feature)
        return x
        