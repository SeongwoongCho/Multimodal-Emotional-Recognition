import albumentations
import numpy as np
import cv2

def get_speech_transform(is_train):
    ## border mode 0 : constant padding, border mode 4 : repeats
    if is_train:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width = 400,border_mode = 0),
                albumentations.RandomCrop(height = 128, width = 400),
                albumentations.RandomBrightnessContrast(brightness_limit = 0.3,contrast_limit = 0.3, p = 0.7),
            ]
        )
    else:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width=400,border_mode = 0),
                albumentations.CenterCrop(height = 128, width = 400),
            ]
        )
    return transform

def sample_frames(x, n_frames = 32,is_train = False):
    # x : C,T,H,W
    T = x.shape[1]
    out = np.zeros(shape = (x.shape[0],n_frames,x.shape[2],x.shape[3]))
    if T<=n_frames:
        out[:,n_frames-T:,:,:] = x
        return out
    
    if is_train == True:
        idxs = np.sort(np.random.choice(x.shape[1],n_frames,replace=False))
    else:
        idxs = np.arange(x.shape[1])[::x.shape[1]//n_frames][:n_frames]
    return x[:,idxs,:,:]


def drop_endframes(x,n_frames = 10):
     return x[:,n_frames:-n_frames,:,:]
    
def normalize(x,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    # x : C,T,H,W
    mean = np.array(mean)[:,np.newaxis,np.newaxis,np.newaxis]
    std = np.array(std)[:,np.newaxis,np.newaxis,np.newaxis]
    return (x/255.-mean)/std

def RandomFlip(x):
    if np.random.uniform()<0.5:
        x = x[:,:,::-1,:]
    if np.random.uniform()<0.5:
        x = x[:,:,:,::-1]
    return x

def video_resize(x,shape):
    # C,T,H,W
    y = []
    for i in range(x.shape[1]):
        y.append(cv2.resize(np.rollaxis(x[:,i,:,:],0,3),shape))
    y = np.array(y) # T,H1,W1,C
    y = np.rollaxis(y,-1,0)
    return y
    
def get_face_transform(is_train=False):
    def face_transform(x):
        x = drop_endframes(x,n_frames = 10)
        x = sample_frames(x,n_frames = 16, is_train=is_train)
#        x = video_resize(x,(112,112))
        if is_train:
            x = RandomFlip(x)
        x = normalize(x)
        return x
    return face_transform

def get_text_transform(is_train):
    return albumentations.PadIfNeeded(min_height = 38, min_width = 200,border_mode = 0)