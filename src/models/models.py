import torch
import torch.nn as nn
from .utils import FrameAttentionModule
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

n_classes = 7
class multimodal_model(nn.Module):
    def __init__(self,speech_model,face_model,text_model):
        super(multimodal_model,self).__init__()
        self.speech_model = speech_model
        self.face_model = face_model
        self.text_model = text_model
        self.multimodal_out = nn.Linear(self.speech_model.feature_dim +
                                        self.face_model.feature_dim +
                                        self.text_model.feature_dim, n_classes)
    
    def forward(self, speech, face, text):
        out_speech, feature_speech = self.speech_model(speech,True)
        out_face, feature_face = self.face_model(face,True)
        out_text, feature_text = self.text_model(text,True)
        out_multimodal = self.multimodal_out(torch.cat([feature_speech,feature_face,feature_text],dim = -1))
        
        return out_multimodal, out_speech, out_face, out_text

def get_text_model(weights = None, nhead = 4, num_layers = 3):
    model = TransformerClassifier(nhead,num_layers)
    if weights:
        model.load_state_dict(torch.load(weights))
    return model

def get_speech_model(coeff = 0, weights = None):
    model = AttnCNN(coeff = coeff)
    if weights:
        model.load_state_dict(torch.load(weights))
    return model

def get_face_model(coeff = 0,weights = None):
#    model = r2plus1d_18(pretrained=True)
    model = CLSTM(coeff = coeff)
    if weights:
        model.load_state_dict(torch.load(weights))
    return model

class CLSTM(nn.Module):
    def __init__(self, coeff = 0):
        super(CLSTM,self).__init__()
        self.backbone_model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.out = AttnBiLSTM(num_classes = n_classes,
                                    input_size = self.backbone_model._fc.in_features,
                                    hidden_size = self.backbone_model._fc.in_features//4)
        self.feature_dim = self.out.feature_dim
        self.backbone_model._fc = nn.Identity()
        
    def forward(self,x,extract_feature=False):
        # x : B,C,T,H,W
        B,C,T,H,W = x.shape
        x = x.transpose(1,2) # B,T,C,H,W
        x = x.reshape(B*T,C,H,W) # B*T,C,H,W
        x = self.backbone_model(x) # B*T,F
        x = x.reshape(B,T,-1) # B,T,F
        output = self.out(x,extract_feature) # B,num_calsses
        return output

class AttnCNN(nn.Module):
    def __init__(self,coeff = 0):
        super(AttnCNN,self).__init__()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b%d'%coeff)
        self.model._conv_stem = Conv2dStaticSamePadding(1,self.model._conv_stem.out_channels, kernel_size = (3,3), stride = (2,2), bias = False, image_size = 128)
        self.out = AttnBiLSTM(num_classes = 7, input_size = self.model._fc.in_features, hidden_size = 512)
        self.feature_dim = self.out.feature_dim
        del self.model._fc
    def forward(self,x,extract_feature = False):
        x = self.model.extract_features(x) # B,C,F,T
        x = torch.mean(x,dim = 2) # B,C,T
        x = x.transpose(1,2)
        x = self.out(x,extract_feature)
        return x

class AttnBiLSTM(nn.Module):
    def __init__(self,num_classes = 7,input_size = 1024, hidden_size = 512):
        super(AttnBiLSTM,self).__init__()
        
        attn_mode = 'relation'
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = nn.LSTM(input_size = input_size,hidden_size=hidden_size,num_layers=1,batch_first = True,bidirectional=True)
        self.attention = FrameAttentionModule(in_channels = hidden_size*2, mode = attn_mode)
        self.dropout = nn.Dropout(0.2)
        self.feature_dim = 4*hidden_size if attn_mode == 'relation' else 2*hidden_size
        self.out = nn.Linear(self.feature_dim,num_classes)
        
    def forward(self,x,extract_feature = False):
        # x : B, T, F
        self.encoder.flatten_parameters()
        output,_ = self.encoder(x) ## B,T,F -> B,T,self.hidden_size * 2
        feature = self.attention(output)
        out = self.dropout(feature)
        out = self.out(out)
        if extract_feature:
            return out,feature
        return out

class TransformerClassifier(nn.Module):
    def __init__(self,nhead,num_layers):
        super(TransformerClassifier,self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(200,num_classes)
        self.feature_dim = 200
    def forward(self,x,extract_feature = False):
        feature = self.transformer_encoder(x)
        feature = torch.mean(feature,dim=1)
        x = self.out(feature)
        if extract_feature:
            return x,feature
        return x