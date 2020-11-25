import librosa
import moviepy.editor as mp
import torch
import cv2
import os
import numpy as np
import scipy
import shutil

from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

class CONFIG():
    FILE_DIR = ''
    SAVE_DIR = ''
    
    TOP_DB = 20
    SAMPLE_RATE = 16000
    HOP_LENGTH = 256
    N_FFT = 512
    N_MELS = 128
    FREQ_RANGE = (80, 8000)
    WINDOW = 'hann'
    
    FACE_RESIZE = 224

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.config = CONFIG()
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
#        self.mtcnn = torch.nn.DataParallel(self.mtcnn)
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
        
        if len(frames[::self.stride]) == 0:
            return []
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                faces.append(np.zeros((self.config.FACE_RESIZE,self.config.FACE_RESIZE,3)))
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return faces    
    
train_config = CONFIG()
val_config = CONFIG()
test_config = CONFIG()
test3_config = CONFIG()

train_config.FILE_DIR = '../qia2020/train/'
train_config.SAVE_DIR = '../features/train/'
val_config.FILE_DIR = '../qia2020/val/'
val_config.SAVE_DIR = '../features/val/'
test_config.FILE_DIR = '../qia2020/test/'
test_config.SAVE_DIR = '../features/test/'
test3_config.FILE_DIR = '../2020-3/test3/'
test3_config.SAVE_DIR = '../features/test3/'

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    keep_all=True,
    device= 'cuda'
)
    
def preprocess_audio(wav,config):
    """
    Args:
        wav: wave
        sr: sampling rate
    Returns:
        input_mels
    """
    
    # get silence interval
    wav,index = librosa.effects.trim(wav,frame_length = config.N_FFT, hop_length = config.HOP_LENGTH,top_db = config.TOP_DB)
    
    # Denoising
    wav[np.argwhere(wav == 0)] = 1e-10
    wav_denoise = scipy.signal.wiener(wav, mysize=None, noise=None)

    # mel spectrogram
    spectrogram = librosa.core.stft(y = wav_denoise, hop_length = CONFIG.HOP_LENGTH, n_fft = config.N_FFT, window=config.WINDOW)
    mel_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=spectrogram,sr=config.SAMPLE_RATE,n_mels=config.N_MELS,
                                            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,fmax=config.FREQ_RANGE[1],fmin=config.FREQ_RANGE[0]))
    
    yS = librosa.util.normalize(mel_spec) ## 각 time step마다 max(abs) 로 normalize (즉 무조건 [-1,1] 범위에 온다.)
    yS = (1+yS)/2 #[0,1]
    
    return yS,index

def preprocess_video(video_list,config):
    if len(video_list) < 350:
        faces_tmp = fast_mtcnn(video_list)
    else:
        faces_tmp = fast_mtcnn(video_list[:350]) + fast_mtcnn(video_list[350:])
    
    faces = []
    for face in faces_tmp:
        try:
            faces.append(cv2.resize(face, (config.FACE_RESIZE, config.FACE_RESIZE)))
        except:
            pass
    
    faces = np.array(faces).swapaxes(1,3).swapaxes(2,3) ## T,size,size,3
    return faces

def preprocess(file,config):
    videoclip = mp.VideoFileClip(os.path.join(config.FILE_DIR,file))
    audio = videoclip.audio.to_soundarray(fps = config.SAMPLE_RATE)[:,0]
    log_mel_spec, index = preprocess_audio(audio,config) ## F,T
    
    n_mel,n_frame = log_mel_spec.shape
    duration = videoclip.duration
    
    start_time = index[0]/config.SAMPLE_RATE
    end_time = index[1]/config.SAMPLE_RATE

    video_list = [videoclip.get_frame(start_time + (end_time-start_time)*i/n_frame) for i in range(n_frame)]
    video_preprocessed = preprocess_video(video_list,config)
    return log_mel_spec, video_preprocessed

def process(config):
    os.makedirs(config.SAVE_DIR + 'speech/',exist_ok = True)
    os.makedirs(config.SAVE_DIR + 'video/',exist_ok = True)
    
    files = [ file for file in os.listdir(config.FILE_DIR) if file.endswith('.mp4')]
    for file in tqdm(files):
        if os.path.exists(os.path.join(config.SAVE_DIR + 'speech/', file[:-4]+'.npy')) and os.path.exists(os.path.join(config.SAVE_DIR + 'speech/', file[:-4]+'.npy')):
            continue
        else:
            try:
                speech_feature, video_feature = preprocess(file,config)
                np.save(os.path.join(config.SAVE_DIR + 'speech/', file[:-4]+'.npy'),speech_feature)
                np.save(os.path.join(config.SAVE_DIR + 'video/', file[:-4]+'.npy'),video_feature)
            except Exception as e:
                print(e)

def process_feature_embedding(config):
    os.makedirs(config.SAVE_DIR + 'video_embedding/',exist_ok = True)
    
    embedding_net = InceptionResnetV1(pretrained='vggface2')
    embedding_net = torch.nn.DataParallel(embedding_net)
    embedding_net.cuda()
    embedding_net.eval()
    files = [file for file in os.listdir(config.SAVE_DIR + 'video/') if file.endswith('.npy')]
    
    for file in tqdm(files):
        inp = torch.Tensor(np.load(config.SAVE_DIR + 'video/' + file)).cuda()
        with torch.no_grad():
            out = embedding_net(inp)
        out = out.cpu().detach().numpy()
        np.save(config.SAVE_DIR + 'video_embedding/' + file, out)

def process_text_embedding(config):
    os.makedirs(config.SAVE_DIR + 'text/', exist_ok = True)
    files = [file for file in os.listdir(config.FILE_DIR) if file.endswith('.npz')]
    for file in tqdm(files):
        shutil.copy(os.path.join(config.FILE_DIR, file), os.path.join(config.SAVE_DIR + 'text/',file))
    
if __name__ == '__main__':
    """
    print("..processing train data")
    process(train_config)

    print("..processing valid data")
    process(val_config)



    print(".. feature processing train data")
    process_feature_embedding(train_config)

    print(".. feature processing valid data")
    process_feature_embedding(val_config)

    
    print(".. text processing train data")
    process_text_embedding(train_config)

    print(".. text processing valid data")
    process_text_embedding(val_config)


    """
    print("..processing test data")
    process(test_config)
    print(".. feature processing test data")
    process_feature_embedding(test_config)
    print(".. text processing test data")
    process_text_embedding(test_config)
    
    print("..processing test3 data")
    process(test3_config)
    print(".. feature processing test data")
    process_feature_embedding(test3_config)
    print("..processing test3 data")
    process_text_embedding(test3_config)