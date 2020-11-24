import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from dataloader.dataset import Dataset
from models.models import *

from tqdm import tqdm
import pandas as pd
import ttach as tta

emo2label = ['hap','ang','dis','fea','sad','neu','sur']

weight_path1 = '../src/logs/multimodal/mixup-lam_2-latest/22184_7_best_1.2906.pth'
weight_path2 = '../src/logs/multimodal/mixup-lam_2-latest-newseed/55460_19_best_1.2704.pth'
weight_path3 = '../src/logs/multimodal/mixup-lam_2-latest-newnewseed/30503_10_best_1.2922.pth'
weight_path4 = '../src/logs/multimodal/mixup-lam_2-seed1234-bigger-models/44368_7_best_1.2946.pth'

test_speech_root_dir = '../features/test/speech'
test_face_root_dir = '../features/test/video'
test_text_root_dir = '../features/test/text'
test_files = [file for file in os.listdir(test_speech_root_dir) if '.npy' in file]
test_files.sort()
test_dataset = Dataset(speech_root_dir = test_speech_root_dir, video_root_dir = test_face_root_dir,text_root_dir = test_text_root_dir,file_list = test_files,label_smoothing = 0,is_train=False, is_test=True, n_frames = 32)
test_loader=data.DataLoader(dataset=test_dataset,batch_size=32,num_workers=20,shuffle=False)

test3_speech_root_dir = '../features/test3/speech'
test3_face_root_dir = '../features/test3/video'
test3_text_root_dir = '../features/test3/text'
test3_files = [file for file in os.listdir(test3_speech_root_dir) if '.npy' in file]
test3_files.sort()
test3_dataset = Dataset(speech_root_dir = test3_speech_root_dir, video_root_dir = test3_face_root_dir,text_root_dir = test3_text_root_dir,file_list = test3_files,label_smoothing = 0,is_train=False, is_test=True, n_frames = 32)
test3_loader=data.DataLoader(dataset=test3_dataset,batch_size=32,num_workers=20,shuffle=False)

speech_model1 = get_speech_model(coeff = 4)
face_model1 = get_face_model(coeff = 0)
text_model1 = get_text_model(nhead = 8, num_layers = 12)
model1 = multimodal_model(speech_model1,face_model1,text_model1)
model1.load_state_dict(torch.load(weight_path1))
model1.cuda()
model1= nn.DataParallel(model1)
model1.eval()

speech_model2 = get_speech_model(coeff = 4)
face_model2 = get_face_model(coeff = 0)
text_model2 = get_text_model(nhead = 8, num_layers = 12)
model2 = multimodal_model(speech_model2,face_model2,text_model2)
model2.load_state_dict(torch.load(weight_path2))
model2.cuda()
model2= nn.DataParallel(model2)
model2.eval()

speech_model3 = get_speech_model(coeff = 4)
face_model3 = get_face_model(coeff = 0)
text_model3 = get_text_model(nhead = 8, num_layers = 12)
model3 = multimodal_model(speech_model3,face_model3,text_model3)
model3.load_state_dict(torch.load(weight_path3))
model3.cuda()
model3= nn.DataParallel(model3)
model3.eval()

speech_model4 = get_speech_model(coeff = 7)
face_model4 = get_face_model(coeff = 1)
text_model4 = get_text_model(nhead = 8, num_layers = 12)
model4 = multimodal_model(speech_model4,face_model4,text_model4)
model4.load_state_dict(torch.load(weight_path4))
model4.cuda()
model4= nn.DataParallel(model4)
model4.eval()

Emotions = []
Emotions3 = []
temperature = 0.1 ## 1 for arithmetic mean, >1 for temparature sharpening, < 1 for temparature smoothing
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
#        tta.Multiply(factors=[0.8, 0.9, 1, 1.1, 1.2]),
        tta.Multiply(factors=[0.9, 1, 1.1])
    ]
)

with torch.no_grad():
    for idx,data in enumerate(tqdm(test_loader)):
        speech = data['speech'].cuda()
        text = data['text'].cuda()
        
        inte_pred = None
        for transformer in transforms:
            face = torch.zeros_like(data['face'])
            for i in range(face.shape[2]):
                face[:,:,i,:,:] = transformer.augment_image(data['face'][:,:,i,:,:])
            face = face.cuda()
            _inte_pred,_,_,_ = model1(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred

            _inte_pred,_,_,_ = model2(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred
   
            _inte_pred,_,_,_ = model3(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred
                
            _inte_pred,_,_,_ = model4(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred
                
        inte_pred = inte_pred / (len(transforms)*4)
        inte_pred = inte_pred.detach().max(1)[1].numpy()
        Emotions.extend([emo2label[pred] for pred in inte_pred])

with torch.no_grad():
    for idx,data in enumerate(tqdm(test3_loader)):
        speech = data['speech'].cuda()
        text = data['text'].cuda()
        
        inte_pred = None
        for transformer in transforms:
            face = torch.zeros_like(data['face'])
            for i in range(face.shape[2]):
                face[:,:,i,:,:] = transformer.augment_image(data['face'][:,:,i,:,:])
            face = face.cuda()
            _inte_pred,_,_,_ = model1(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred

            _inte_pred,_,_,_ = model2(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred

            _inte_pred,_,_,_ = model3(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred

            _inte_pred,_,_,_ = model4(speech,face,text)
            
            _inte_pred = F.softmax(_inte_pred,dim = -1).pow(temperature).cpu().detach()
            _inte_pred = _inte_pred / torch.sum(_inte_pred, dim = -1, keepdim = True)
            if inte_pred is None:
                inte_pred = _inte_pred
            else:
                inte_pred += _inte_pred                
                
        inte_pred = inte_pred / (len(transforms)*4)
        inte_pred = inte_pred.detach().max(1)[1].numpy()
        Emotions3.extend([emo2label[pred] for pred in inte_pred])


df1 = pd.read_csv('../features/test1.csv')
df2 = pd.read_csv('../features/test2.csv')
df3 = pd.read_csv('../features/test3.csv')

FileIDs1 = df1['FileID'].tolist()
_Emotions1 = []
FileIDs2 = df2['FileID'].tolist()
_Emotions2 = []
FileIDs3 = df3['FileID'].tolist()
_Emotions3 = []
for file in FileIDs1:
    emotion = 'neu'
    for idx,_file in enumerate(test_files):
        if int(_file.split('-')[0]) == file:
            emotion = Emotions[idx]
    _Emotions1.append(emotion)
    
for file in FileIDs2:
    emotion = 'neu'
    for idx,_file in enumerate(test_files):
        if int(_file.split('-')[0]) == file:
            emotion = Emotions[idx]
    _Emotions2.append(emotion)

for file in FileIDs3:
    emotion = 'neu'
    for idx,_file in enumerate(test3_files):
        if int(_file.split('-')[0]) == file:
            emotion = Emotions3[idx]
    _Emotions3.append(emotion)

df1['Emotion'] = _Emotions1
df2['Emotion'] = _Emotions2
df3['Emotion'] = _Emotions3
    
df1.to_csv('./test1_submission.csv',index=False)
df2.to_csv('./test2_submission.csv',index=False)
df3.to_csv('./test3_submission.csv',index=False)