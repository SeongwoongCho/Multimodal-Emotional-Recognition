from utils.utils import *
from utils.SpecAugment import spec_augment_pytorch
from torch.utils import data
from .transform import get_speech_transform, get_face_transform

seed_everything(42)
label2emo={'hap':0,'ang':1,'dis':2,'fea':3,'sad':4,'neu':5,'sur':6}

def parse_label(fileName,smooth_weight):
    file = fileName.split('.')[0].split('-')
    inte_label, face_label, speech_label = file[-3], file[-2], file[-1]

    inte_label = label2emo[inte_label]
    inte_label = to_onehot(label=inte_label, num_classes = 7)

    face_label = label2emo[face_label]
    face_label = to_onehot(label=face_label, num_classes = 7)

    speech_label = label2emo[speech_label]
    speech_label = to_onehot(label=speech_label, num_classes = 7)

    inte_label = (1-smooth_weight)*inte_label + smooth_weight*(1-inte_label)/6
    face_label = (1-smooth_weight)*face_label + smooth_weight*(1-face_label)/6
    speech_label = (1-smooth_weight)*speech_label + smooth_weight*(1-speech_label)/6

    inte_label = inte_label.astype('float32')
    face_label = face_label.astype('float32')
    speech_label = speech_label.astype('float32')

    return inte_label, face_label, speech_label
    
class Dataset(data.Dataset):
    def __init__(self,speech_root_dir= None,video_root_dir= None,text_root_dir = None, file_list= None, label_smoothing = 0,is_train=True):
        self.file_list = file_list
        self.speech_root_dir = speech_root_dir
        self.text_root_dir = text_root_dir
        self.video_root_dir = video_root_dir
        
        self.smooth_weight = label_smoothing
        self.is_train=is_train
        self.speech_transform = get_speech_transform(is_train)
        self.face_transform = get_face_transform(is_train)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,idx):
        ## label load and preprocess
        fileName = self.file_list[idx]
        inte_label, face_label, speech_label = parse_label(fileName,self.smooth_weight)
        
        ## input load and preprocess
        data = {'inte_label' : torch.from_numpy(inte_label),'speech_label' : torch.from_numpy(speech_label),'face_label' : torch.from_numpy(face_label)}
        
        ## load speech
        if self.speech_root_dir is not None:
            yS = np.load(os.path.join(self.speech_root_dir,fileName)).astype('float32')[...,np.newaxis] # F,T,1
            yS = self.speech_transform(image = yS)['image'] ## F,T,1
            yS = np.rollaxis(yS,-1,0) # 1,F,T
            yS = torch.from_numpy(yS)
            if self.is_train:
                yS = spec_augment_pytorch.spec_augment(mel_spectrogram = yS, time_warping_para=120, frequency_masking_para=27,
                     time_masking_para=50, frequency_mask_num=3, time_mask_num=4)
            data['speech'] = yS
        if self.video_root_dir is not None:
            face = np.load(os.path.join(self.video_root_dir,fileName)).astype('float32')[...,np.newaxis] ## T, F,1 
            face = self.face_transform(image = face)['image'] # -> 여기도 패딩을 넣어줘야한다.
            face = np.rollaxis(face,-1,0) # 1,T,F
            data['face'] = torch.from_numpy(face).transpose(1,2) ## 1,F,T
        
        if self.text_root_dir is not None:
            text = np.load(os.path.join(self.text_root_dir,fileName)).astype('float32')
            data['text'] = torch.from_numpy(text)
           
        return data