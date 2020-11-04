from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from .attention import attention_bilstm,attention_CNNBilstm
import torch.nn as nn
import torch

n_classes = 7


def get_speech_model(coeff = 0, weights = None,hidden_size = 512):
    model = attentioned_CNN(coeff = coeff,hidden_size = hidden_size)
#    model = vanilla_CNN(coeff = coeff)
    if weights:
        model.load_state_dict(torch.load(weights))

    return model

class vanilla_CNN(nn.Module):
    def __init__(self,coeff = 0):
        super(vanilla_CNN,self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.model._conv_stem = Conv2dStaticSamePadding(1,self.model._conv_stem.out_channels, kernel_size = (3,3), stride = (2,2), bias = False, image_size = 128)
        self.model._fc = nn.Linear(self.model._fc.in_features,7)
#        self.out = attention_bilstm(num_classes = 7, input_size = self.model._fc.in_features, hidden_size = hidden_size)
#        del self.model._fc
    def forward(self,x,extract_feature = False):
        x = self.model.extract_features(x) 
        x = self.model._avg_pooling(x)
        feature = x.flatten(start_dim=1)
        if not extract_feature:
            x = self.model._dropout(feature)
            x = self.model._fc(feature)
            return x
        return x,feature

class attentioned_CNN(nn.Module):
    def __init__(self,coeff = 0,hidden_size = 512):
        super(attentioned_CNN,self).__init__()
        self.hidden_size= hidden_size
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.model._conv_stem = Conv2dStaticSamePadding(1,self.model._conv_stem.out_channels, kernel_size = (3,3), stride = (2,2), bias = False, image_size = 128)
        self.out = attention_bilstm(num_classes = 7, input_size = self.model._fc.in_features, hidden_size = hidden_size)
        del self.model._fc
    def forward(self,x,extract_feature = False):
        x = self.model.extract_features(x) # B,C,F,T
        x = torch.mean(x,dim = 2) # B,C,T
        x = x.transpose(1,2)
        x = self.out(x,extract_feature)
        return x

class multimodal_model(nn.Module):
    def __init__(self,speech_model_coeff = 0,hidden_sizes= [512,256]):
        super(multimodal_model,self).__init__()
        self.speech_model = get_speech_model(coeff = speech_model_coeff)
        self.face_model = attention_CNNBilstm(num_classes = n_classes, input_size = 512, hidden_size = hidden_sizes[0])
        self.text_model = attention_CNNBilstm(num_classes = n_classes, input_size = 200, hidden_size = hidden_sizes[1])
        
        self.multimodal_out = nn.Linear(2*(self.speech_model.hidden_size + hidden_sizes[0] + hidden_sizes[1]), n_classes)
        
    def forward(self,speech, face, text):
        out_speech, feature_speech = self.speech_model(speech,True)
        out_face, feature_face = self.face_model(face,True)
        out_text, feature_text = self.text_model(text,True)
        
        out_multimodal = self.multimodal_out(torch.cat([feature_speech,feature_face,feature_text],dim = -1))
        
        return out_multimodal, out_speech, out_face, out_text