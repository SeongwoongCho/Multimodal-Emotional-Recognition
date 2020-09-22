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
    
class SpeechDataset(data.Dataset):
    def __init__(self,root_dir,file_list,label_smoothing = 0,is_train=True):
        self.file_list = file_list
        self.root_dir = root_dir
        self.smooth_weight = label_smoothing
        self.is_train=is_train
        self.transform = get_speech_transform(is_train)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,idx):
        ## label load and preprocess
        fileName = self.file_list[idx]
        inte_label, face_label, speech_label = parse_label(fileName,self.smooth_weight)
        
        ## input load and preprocess
        yS = np.load(os.path.join(self.root_dir,fileName)).astype('float32')[...,np.newaxis] # F,T,1
        yS = self.transform(image = yS)['image'] ## F,T,1
        yS = np.rollaxis(yS,-1,0)
        yS = torch.from_numpy(yS)
        if self.is_train:
            speech = spec_augment_pytorch.spec_augment(mel_spectrogram = yS, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1)
        return {'speech' : yS , 'speech_label' : torch.from_numpy(speech_label)}


def get_speech_collater(is_train):
    def collater(data):
        speech = torch.cat([s['speech'].unsqueeze(0) for s in data],dim = 0)
        speech_label = torch.cat([s['speech_label'].unsqueeze(0) for s in data],dim = 0)
        return {'speech': speech, 'speech_label': speech_label}
    return collater