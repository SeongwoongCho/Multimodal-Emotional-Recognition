import torch
import torch.nn as nn

from torch.optim import AdamW,SGD
from torch.utils import data

from utils.utils import *
from models.model_speech import get_speech_model
from models.model_face import get_face_model
from dataloader.dataset import SpeechDataset,get_speech_collater
seed_everything(42)

import numpy as np
import os
import time
import csv
import warnings

# from apex import amp
from tqdm import tqdm
from importify import Serializable

class Config(Serializable):
    def __init__(self):
        super(Config, self).__init__()
        
        ## training mode
        self.mode = 'speech'
        self.exp_name = 'baseline'
        assert self.mode in ['speech','face','multi']
        
        ## training parameters
        self.learning_rate = 4e-3
        self.batch_size = 32
        self.n_epoch = 100
        self.optim = 'adamw'
        self.weight_decay = 1e-5
        self.num_workers = 16
        
        ## model parameters
        self.load_weights = None
        self.coeff = 0
        
        ## other hyper parameters
        self.mixup_prob = 1.
        self.mixup_alpha = 0.5
        self.label_smoothing = 0.1
        
        ## training_options
        self.amp = False
        

config = Config()
config.parse()

if config.amp:
    from apex import amp

save_path = './logs/{}/{}'.format(config.mode,config.exp_name)
os.makedirs(save_path,exist_ok=True)
saved_status = config.export_json(path=os.path.join(save_path,'saved_status.json'))
print(saved_status)

train_speech_root_dir = '../features/train/speech'
train_face_root_dir = '../features/train/video_embedding'
valid_speech_root_dir = '../features/val/speech'
valid_face_root_dir = '../features/val/video_embedding'

train_speech_files = [file for file in os.listdir(train_speech_root_dir) if '.npy' in file]
train_face_files = [file for file in os.listdir(train_face_root_dir) if '.npy' in file]
valid_speech_files = [file for file in os.listdir(valid_speech_root_dir) if '.npy' in file]
valid_face_files = [file for file in os.listdir(valid_face_root_dir) if '.npy' in file]

if config.mode == 'speech':
    train_dataset = SpeechDataset(root_dir = train_speech_root_dir,file_list = train_speech_files,label_smoothing = config.label_smoothing,is_train=True)
    valid_dataset = SpeechDataset(root_dir = valid_speech_root_dir,file_list = valid_speech_files,label_smoothing = 0,is_train=False)
    train_loader=data.DataLoader(dataset=train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True,collate_fn = get_speech_collater(is_train = True))
    valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=False, collate_fn = get_speech_collater(is_train = False))
else:
    raise NotImplemented

if config.mode == 'speech' or config.mode == 'face':
    model = get_speech_model(coeff = config.coeff, weights = config.load_weights)
    model.cuda()
    
    if not config.amp:
        model= nn.DataParallel(model)
    if config.optim == 'adamw':
        optimizer = AdamW(model.parameters(),lr=config.learning_rate,weight_decay = config.weight_decay)

elif config.mode == 'multi':
    raise NotImplemented

if config.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model= nn.DataParallel(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = config.n_epoch*len(train_loader))
criterion = cross_entropy()

if config.mode == 'speech' or 'face':
    best_acc=np.inf
    iter = 0
    for epoch in range(config.n_epoch):
        train_loss=0
        optimizer.zero_grad()
        model.train()
        for idx,data in enumerate(tqdm(train_loader)):
            speech = data['speech'].cuda()
            speech_label = data['speech_label'].cuda()
            if np.random.uniform(0,1) < config.mixup_prob:
                speech,speech_label_a,speech_label_b,lam = mixup_data(speech,speech_label,config.mixup_alpha)
                pred = model(speech)
                loss = criterion(pred,speech_label_a*lam + speech_label_b*(1-lam))
            else:
                pred = model(speech)
                loss = criterion(pred,speech_label)
            if config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss+=loss.item()/len(train_loader)
            scheduler.step(iter)
            iter +=1

        valid_loss=0
        valid_acc=0
        model.eval()
        for idx,data in enumerate(tqdm(valid_loader)):
            speech = data['speech'].cuda()
            speech_label = data['speech_label'].cuda()
            with torch.no_grad():
                pred = model(speech)
                loss = criterion(pred,speech_label)
            valid_loss+=loss.item()/len(valid_loader)
            pred=pred.detach().max(1)[1]
            speech_label = speech_label.detach().max(1)[1]
            valid_acc+=pred.eq(speech_label.view_as(pred)).sum().item()
        valid_acc/=len(valid_list)

        torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
        print("Epoch [%d]/[%d] train_loss: %.6f valid_loss: %.6f valid_acc:%.6f"%(
        epoch,n_epoch,train_loss,valid_loss,valid_acc))
