import torch
import torch.nn as nn

from torch.optim import AdamW,SGD
from torch.utils import data

from utils.utils import *
from utils.warmup_scheduler import GradualWarmupScheduler
from models.model import get_speech_model, multimodal_model
from models.attention import attention_bilstm,attention_CNNBilstm
from dataloader.dataset import Dataset
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
        assert self.mode in ['speech','face','text','multimodal']
        
        ## training parameters
        self.learning_rate = 4e-3
        self.batch_size = 32
        self.n_epoch = 100
        self.optim = 'adamw'
        self.weight_decay = 1e-5
        self.num_workers = 16
        self.warmup = 1
        
        ## model parameters
        self.speech_load_weights = 'default'
        self.text_load_weights = 'default'
        self.face_load_weights = 'default'
        
        self.coeff = 0
        self.freeze_backbone = False
        
        ## other hyper parameters
        self.mixup_prob = 1.
        self.mixup_alpha = 0.5
        self.label_smoothing = 0.1
        self.lam = 0.5
        
        ## training_options
        self.amp = False
        
config = Config()
config.parse()

if config.amp:
    from apex import amp

save_path = './logs/{}/{}'.format(config.mode,config.exp_name)
os.makedirs(save_path,exist_ok=True)
saved_status = config.export_json(path=os.path.join(save_path,'saved_status.json'))
if config.speech_load_weights == 'default':
    config.speech_load_weights = None
if config.face_load_weights == 'default':
    config.face_load_weights = None
if config.text_load_weights == 'default':
    config.text_load_weights = None

train_speech_root_dir = '../features/train/speech'
train_face_root_dir = '../features/train/video_embedding'
train_text_root_dir = '../features/train/text'
valid_speech_root_dir = '../features/val/speech'
valid_face_root_dir = '../features/val/video_embedding'
valid_text_root_dir = '../features/val/text'
train_files = [file for file in os.listdir(train_speech_root_dir) if '.npy' in file]
valid_files = [file for file in os.listdir(valid_speech_root_dir) if '.npy' in file]

train_dataset = Dataset(speech_root_dir = train_speech_root_dir, video_root_dir = train_face_root_dir, text_root_dir = train_text_root_dir, file_list = train_files,label_smoothing = config.label_smoothing,is_train=True)
valid_dataset = Dataset(speech_root_dir = valid_speech_root_dir, video_root_dir = valid_face_root_dir,text_root_dir = valid_text_root_dir,file_list = valid_files,label_smoothing = 0,is_train=False)
train_loader=data.DataLoader(dataset=train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True)
valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=False)

if config.mode == 'speech':
    model = get_speech_model(coeff = config.coeff, weights = config.speech_load_weights)
    model.cuda()
elif config.mode == 'face':
    model = attention_CNNBilstm(num_classes = 7, input_size = 512, hidden_size = 256)
    model.cuda()
elif config.mode == 'text':
    model = attention_CNNBilstm(num_classes = 7, input_size = 200, hidden_size = 100)
    model.cuda()
elif config.mode == 'multimodal':
    assert config.speech_load_weights is not None
    if config.freeze_backbone:
        for param in model.model.parameters():
            param.requires_grad = False
    model = multimodal_model(config.coeff, hidden_sizes = [256,100])
    model.speech_model.load_state_dict(torch.load(config.speech_load_weights))
#    model.face_model.load_state_dict(torch.load(config.face_load_weights))
#    model.text_model.load_state_dict(torch.load(config.text_load_weights))
    model.cuda()
    
if not config.amp:
    model= nn.DataParallel(model)
        
if config.optim == 'adamw':
    optimizer = AdamW(model.parameters(),lr=config.learning_rate,weight_decay = config.weight_decay)
elif config.optim == 'sgd':
    optimizer = SGD(model.parameters(),lr=config.learning_rate,weight_decay = config.weight_decay, momentum = 0.9, nesterov = True)

if config.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model= nn.DataParallel(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = config.n_epoch*len(train_loader))
if config.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier = 1, total_epoch = config.warmup*len(train_loader), after_scheduler = scheduler)
    
criterion = cross_entropy()

if config.mode == 'multimodal':
    best_acc=np.inf
    step = 0
    for epoch in range(config.n_epoch):
        train_loss = 0
        train_inte_loss = 0
        train_face_loss = 0
        train_speech_loss = 0
        
        optimizer.zero_grad()
        model.train()
        
        progress_bar = tqdm(train_loader)
        for idx,data in enumerate(progress_bar):
            speech = data['speech'].cuda()
            face = data['face'].cuda()
            text = data['text'].cuda()
            
            speech_label = data['speech_label'].cuda()
            face_label = data['face_label'].cuda()
            inte_label = data['inte_label'].cuda()
            
            if np.random.uniform(0,1) < config.mixup_prob:
                speech,face,text,speech_label_a,speech_label_b,face_label_a,face_label_b,inte_label_a,inte_label_b,lam = mixup_datas(speech,face,text,speech_label,face_label,inte_label,config.mixup_alpha)
                inte_pred, speech_pred, face_pred, _ = model(speech,face,text)
                
                inte_loss = criterion(inte_pred, inte_label_a*lam + inte_label_b*(1-lam))
                speech_loss = criterion(speech_pred, speech_label_a*lam + speech_label_b*(1-lam))
                face_loss = criterion(face_pred, face_label_a*lam + face_label_b*(1-lam))
            else:
                inte_pred, speech_pred, face_pred, _ = model(speech,face,text)
                inte_loss = criterion(inte_pred, inte_label)
                speech_loss = criterion(speech_pred, speech_label)
                face_loss = criterion(face_pred, face_label)
            
            loss = inte_loss + config.lam*(speech_loss + face_loss)/2
            
            if config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss+=loss.item()/len(train_loader)
            train_inte_loss += inte_loss.item()/len(train_loader)
            train_face_loss += face_loss.item()/len(train_loader)
            train_speech_loss += speech_loss.item()/len(train_loader)
            
            scheduler.step(step)
            step +=1
            
            progress_bar.set_description(
                'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. inte_loss: {:.5f} speech_loss: {:.5f} face_loss: {:.5f} Total loss: {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, config.n_epoch, idx + 1, len(train_loader), inte_loss.item(), speech_loss.item(), face_loss.item(), loss.item()))
        
        valid_loss=0
        valid_inte_loss = 0
        valid_face_loss = 0
        valid_speech_loss = 0
        
        valid_acc=0
        model.eval()
        for idx,data in enumerate(tqdm(valid_loader)):
            speech = data['speech'].cuda()
            face = data['face'].cuda()
            text = data['text'].cuda()
            
            speech_label = data['speech_label'].cuda()
            face_label = data['face_label'].cuda()
            inte_label = data['inte_label'].cuda()
            
            with torch.no_grad():
                inte_pred, speech_pred, face_pred, _ = model(speech,face,text)
                inte_loss = criterion(inte_pred, inte_label)
                speech_loss = criterion(speech_pred, speech_label)
                face_loss = criterion(face_pred, face_label)

                loss = inte_loss + config.lam *(speech_loss + face_loss)/2
            
            valid_loss+=loss.item()/len(valid_loader)
            valid_inte_loss += inte_loss.item()/len(valid_loader)
            valid_face_loss += face_loss.item()/len(valid_loader)
            valid_speech_loss += speech_loss.item()/len(valid_loader)
            
            inte_pred=inte_pred.detach().max(1)[1]
            inte_label = inte_label.detach().max(1)[1]
            acc = inte_pred.eq(inte_label.view_as(inte_pred)).sum().item() / len(inte_pred)
            valid_acc+=acc/len(valid_loader)

        torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_inte_loss)))
        print("Epoch [%d]/[%d] train_loss: %.6f train_inte_loss: %.6f train_speech_loss: %.6f train_face_loss: %.6f valid_loss: %.6f valid_inte_loss: %.6f valid_speech_loss: %.6f valid_face_loss: %.6f valid_acc: %.6f"%(epoch,config.n_epoch,train_loss,train_inte_loss, train_speech_loss, train_face_loss, valid_loss,valid_inte_loss, valid_speech_loss, valid_face_loss, valid_acc))

if config.mode == 'face':
    best_acc=np.inf
    step = 0
    for epoch in range(config.n_epoch):
        train_loss=0
        optimizer.zero_grad()
        model.train()
        
        progress_bar = tqdm(train_loader)
        for idx,data in enumerate(progress_bar):
            face = data['face'].cuda()
            face_label = data['face_label'].cuda()
            
            if np.random.uniform(0,1) < config.mixup_prob:
                face,face_label_a,face_label_b,lam = mixup_data(face,face_label,config.mixup_alpha)
                pred = model(face)
                loss = criterion(pred,face_label_a*lam + face_label_b*(1-lam))
            else:
                pred = model(face)
                loss = criterion(pred,face_label)
            if config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss+=loss.item()/len(train_loader)
            scheduler.step(step)
            step +=1
            progress_bar.set_description(
                'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. current loss: {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, config.n_epoch, idx + 1, len(train_loader), loss.item()))
        
        valid_loss=0
        valid_acc=0
        model.eval()
        for idx,data in enumerate(tqdm(valid_loader)):
            face = data['face'].cuda()
            face_label = data['face_label'].cuda()
            with torch.no_grad():
                pred = model(face)
                loss = criterion(pred,face_label)
            valid_loss+=loss.item()/len(valid_loader)
            pred=pred.detach().max(1)[1]
            face_label = face_label.detach().max(1)[1]
            acc = pred.eq(face_label.view_as(pred)).sum().item() / len(pred)
            valid_acc+=acc/len(valid_loader)

        torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
        print("Epoch [%d]/[%d] train_loss: %.6f valid_loss: %.6f valid_acc:%.6f"%(
        epoch,config.n_epoch,train_loss,valid_loss,valid_acc))

if config.mode == 'text':
    best_acc=np.inf
    step = 0
    for epoch in range(config.n_epoch):
        train_loss=0
        optimizer.zero_grad()
        model.train()
        
        progress_bar = tqdm(train_loader)
        for idx,data in enumerate(progress_bar):
            text = data['text'].cuda()
            text_label = data['inte_label'].cuda()
            if np.random.uniform(0,1) < config.mixup_prob:
                text,text_label_a,text_label_b,lam = mixup_data(text,text_label,config.mixup_alpha)
                pred = model(text)
                loss = criterion(pred,text_label_a*lam + text_label_b*(1-lam))
            else:
                pred = model(text)
                loss = criterion(pred,text_label)
            if config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss+=loss.item()/len(train_loader)
            scheduler.step(step)
            step +=1
            progress_bar.set_description(
                'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. current loss: {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, config.n_epoch, idx + 1, len(train_loader), loss.item()))
        
        valid_loss=0
        valid_acc=0
        model.eval()
        for idx,data in enumerate(tqdm(valid_loader)):
            text = data['text'].cuda()
            text_label = data['inte_label'].cuda()
            with torch.no_grad():
                pred = model(text)
                loss = criterion(pred,text_label)
            valid_loss+=loss.item()/len(valid_loader)
            pred=pred.detach().max(1)[1]
            text_label = text_label.detach().max(1)[1]
            acc = pred.eq(text_label.view_as(pred)).sum().item() / len(pred)
            valid_acc+=acc/len(valid_loader)

        torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
        print("Epoch [%d]/[%d] train_loss: %.6f valid_loss: %.6f valid_acc:%.6f"%(
        epoch,config.n_epoch,train_loss,valid_loss,valid_acc))        
        
if config.mode == 'speech':
    best_acc=np.inf
    step = 0
    for epoch in range(config.n_epoch):
        train_loss=0
        optimizer.zero_grad()
        model.train()
        
        progress_bar = tqdm(train_loader)
        for idx,data in enumerate(progress_bar):
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
            scheduler.step(step)
            step +=1
            progress_bar.set_description(
                'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. current loss: {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, config.n_epoch, idx + 1, len(train_loader), loss.item()))
        
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
            acc = pred.eq(speech_label.view_as(pred)).sum().item() / len(pred)
            valid_acc+=acc/len(valid_loader)

        torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
        print("Epoch [%d]/[%d] train_loss: %.6f valid_loss: %.6f valid_acc:%.6f"%(
        epoch,config.n_epoch,train_loss,valid_loss,valid_acc))
