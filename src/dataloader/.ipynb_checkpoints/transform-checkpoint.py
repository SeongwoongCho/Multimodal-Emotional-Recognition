import albumentations
import numpy as np
import cv2
import numpy as np
import skimage.transform
import albumentations.augmentations.functional as F

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

def get_face_transform(is_train=False, n_frames = 16):
    ran_rotate = get_video_random_rotate(30, prob = 0.7)
#    rotate = get_video_random_rot90()
#    flip = get_video_random_flip(prob = 0.5)
    bc = get_video_random_brightness_contrast(brightness = 0.5, contrast = 0.5, prob = 1 )
    def face_transform(x):
        x = drop_endframes(x,n_frames = 10)
        x = sample_frames(x,n_frames = n_frames, is_train=is_train).astype('uint8')
        if is_train:
            # C,T,H,W -> T,H,W,C
            if np.random.uniform()<0.5:
                x = video_horizontal_flip(x)
            x = bc(x)
#            x = rotate(x)
            x = ran_rotate(x)
        x = normalize(x)
        return x
    return face_transform

def get_text_transform(is_train):
    return albumentations.PadIfNeeded(min_height = 38, min_width = 200,border_mode = 0)

def get_video_random_rot90():
    def video_random_rotate(vid):
        a = np.random.choice([0,90,180,270],1)[0]
        vid = video_rotate(vid,a)
        return vid
    return video_random_rotate
def get_video_random_rotate(angle, prob = 1):
    def video_random_rotate(vid):
        if np.random.uniform() < prob:
            a = np.random.uniform(-angle,angle)
            vid = video_rotate(vid,a)
        return vid
    return video_random_rotate

def get_video_random_flip(prob = 0.5):
    def video_random_flip(vid):
        if np.random.uniform() < prob:
            vid = video_horizontal_flip(vid)
        if np.random.uniform() < prob:
            vid = video_vertical_flip(vid)
        return vid
    return video_random_flip

def get_video_random_brightness_contrast(brightness = 0.5, contrast = 0.5, prob = 0.5):
    def video_random_brightness_contrast(vid):
        if np.random.uniform() < prob:
            alpha = 1.0 + np.random.uniform(-contrast,contrast),
            beta = 0.0 + np.random.uniform(-brightness, brightness)
            return video_brightness_contrast(vid,alpha,beta)
        return vid
    return video_random_brightness_contrast

def video_rotate(vid, angle):
    if angle == 0:
        return vid
    transformed = np.zeros_like(vid).astype('uint8')
    for i in range(vid.shape[1]):
        transformed[:,i,:,:] = (skimage.transform.rotate(vid[:,i,:,:].transpose(1,2,0),angle, mode = 'constant')*255).astype('uint8').transpose(2,0,1)
    return transformed

def video_horizontal_flip(vid):
    return vid[:,:,:,::-1]

def video_vertical_flip(vid):
    return vid[:,:,::-1,:]

def video_brightness_contrast(vid, alpha, beta):
    # C,T,H,W
    transformed = np.zeros_like(vid).astype('uint8')
    for i in range(vid.shape[1]):
        transformed[:,i,:,:] = F.brightness_contrast_adjust(vid[:,i,:,:].transpose(1,2,0), alpha, beta).transpose(2,0,1)
    return transformed

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