from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from .attention import attention_bilstm
import torch.nn as nn
import torch

n_classes = 7


def get_speech_model(coeff = 0, weights = None):
    model = attentioned_CNN(coeff = coeff)
    if weights:
        model.load_state_dict(torch.load(weights))

    return model

class attentioned_CNN(nn.Module):
    def __init__(self,coeff = 0):
        super(attentioned_CNN,self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.model._conv_stem = Conv2dStaticSamePadding(1,self.model._conv_stem.out_channels, kernel_size = (3,3), stride = (2,2), bias = False, image_size = 128)
        self.out = attention_bilstm(num_classes = 7, input_size = self.model._fc.in_features, hidden_size = 512)
        del self.model._fc
    def forward(self,x,extract_feature = False):
        x = self.model.extract_features(x) # B,C,F,T
        x = torch.mean(x,dim = 2) # B,C,T
        x = x.transpose(1,2)
        x = self.out(x,extract_feature)
        return x

class multimodal_model(nn.Module):
    def __init__(self,speech_model = None):
        super(multimodal_model,self).__init__()
        self.speech_model = speech_model
        self.face_model = attention_bilstm(num_classes = n_classes, input_size = 512, hidden_size = 256)
        self.text_model = attention_bilstm(num_classes = n_classes, input_size = 200, hidden_size = 100)
        
        self.multimodal_out = nn.Linear(2*(speech_model.out.hidden_size + 256 + 100), n_classes)
        
    def forward(self,speech, face, text):
        out_speech, feature_speech = self.speech_model(speech,True)
        out_face, feature_face = self.face_model(face,True)
        out_text, feature_text = self.text_model(text,True)
        
        out_multimodal = self.multimodal_out(torch.cat([feature_speech,feature_face,feature_text],dim = -1))
        
        return out_multimodal, out_speech, out_face, out_text