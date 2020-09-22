from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import torch.nn as nn
import torch

n_classes = 7

def get_speech_model(coeff = 0, weights = None):
    model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
    model._conv_stem = Conv2dStaticSamePadding(1,32, kernel_size = (3,3), stride = (2,2), bias = False,image_size = 128)
    model._fc = nn.Linear(model._fc.in_features,n_classes)
    if weights:
        model.load_state_dict(torch.load(weights)['model_state_dict'])
    
    return model