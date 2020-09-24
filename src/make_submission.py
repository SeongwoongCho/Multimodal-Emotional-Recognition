import os
import torch
import torch.nn as nn
from torch.utils import data
from dataloader.dataset import Dataset
from models.model import get_speech_model, multimodal_model
from tqdm import tqdm
import pandas as pd

emo2label = ['hap','ang','dis','fea','sad','neu','sur']

weight_path = '../src/logs/multimodal/baseline/87_best_1.3893.pth'

test_speech_root_dir = '../features/test/speech'
test_face_root_dir = '../features/test/video_embedding'
test_text_root_dir = '../features/test/text'

test_files = [file for file in os.listdir(test_speech_root_dir) if '.npy' in file]

test_dataset = Dataset(speech_root_dir = test_speech_root_dir, video_root_dir = test_face_root_dir,text_root_dir = test_text_root_dir,file_list = test_files,label_smoothing = 0,is_train=False, is_test=True)
test_loader=data.DataLoader(dataset=test_dataset,batch_size=512,num_workers=20,shuffle=False)

model = get_speech_model(coeff = 4)
model = multimodal_model(model)
model.load_state_dict(torch.load(weight_path))
model.cuda()

model= nn.DataParallel(model)

Emotions = []

for idx,data in enumerate(tqdm(test_loader)):
    with torch.no_grad():
        speech = data['speech'].cuda()
        face = data['face'].cuda()
        text = data['text'].cuda()

        inte_pred, speech_pred, face_pred, _ = model(speech,face,text)
        inte_pred=inte_pred.detach().max(1)[1].cpu().detach().numpy()
        
        Emotions.extend([emo2label[pred] for pred in inte_pred])

FileIDs = [int(file.split('-')[0]) for file in test_files] 

df = pd.DataFrame({'FileID' : FileIDs,'Emotion':Emotions})
df.to_csv('./submission.csv',index=False)