# Multimodal-Emotional-Recognition

# Dataset
- QIA2020/MERC2020
    - video / word2vec embedding of sentence is given!
    - labels for each attribute(face,speech,integrate)
    - The target task is maximizing the accuracy of integrate-emotion-label(multimodal label)
    - 7 emotions! 'hap','ang','dis','fea','sad','neu','sur'
- You can train on your own dataset by 

# directory structure

```
/
├── features/
│   ├── train/
│   │   ├── speech/
│   │   ├── text/
│   │   ├── video/
│   │   └── video_embedding/
│   ├── val/ 
│   │   ├── speech/
│   │   ├── text/
│   │   ├── video/
│   │   └── video_embedding/
│   └── test/
│   │   ├── speech/*.npy
│   │   ├── text/*.npz
│   │   ├── video/*.npy
│   │   └── video_embedding/*.npy
│   └── test_merc/
│       ├── speech/*.npy
│       ├── text/*.npz
│       ├── video/*.npy
│       └── video_embedding/*.npy
├── qia2020/
│   ├── train/
│   ├── val/ 
│   └── test/
├── merc2020/
│   ├── test1/
│   ├── test2/ 
│   └── test3/
└── src/
    ├── preprocess.py
    ├── models/
    │   └── models.py
    └── utils/
        └── utils.py
```

# Resources
- RTX6000 * 2
- RAM 128G
- supported by NIPA

# How to start

```
$ python3 preprocess.py

$ python train.py --mode speech --exp_name AttnCNN-B4 --learning_rate 4e-3 --batch_size 256 --n_epoch 100 --optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --mixup_prob 1 --mixup_alpha 1 --label_smoothing 0.1 --amp True >> logs.txt

$ python train.py --mode face --exp_name CLSTM-B0 --learning_rate 4e-3 --batch_size 32 --n_epoch 20 --optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --face_coeff 0 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.1 --amp True >> logs.txt

$ python train.py --mode text --exp_name transformer-3_4 --learning_rate 4e-3 --batch_size 128 --n_epoch 20 --optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --face_coeff 0 --cutmix_prob 1 --cutmix_beta 1 --label_smoothing 0.1 --amp True >> logs.txt

$ python3 train.py --mode multimodal --exp_name 201111 --speech_load_weights {path1} --text_load_weights {path} --face_load_weights {path3} --learning_rate 4e-3 --batch_size 32 --n_epoch 3 --optim sgd --warmup 0 --weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --face_coeff 0 --lam 0 --label_smoothing 0.1 --amp True >> logs.txt
```

# Preprocessing
- The preprocessing is optimized for the "QIA(Qualcomm innovation awards)2020/MERC(multimodal-emotional-recognition-challenge)2020" dataset.

Speech -> denoising / Remove Silence / Log-scale Mel spectrogram
Video -> Frame-level FaceCrop(MTCNN)/Resize
Text -> word2vec embedding of the each word of a sentence (This is given on the QIA2020/MERC2020 challenge!)

# Augmentation
- Speechmodel
    - RandomClip
    - RandomBrightnessContrast
    - SpecAug
    - Mixup
- Facemodel
    - RandomFrameSelection
    - RandomFlip
    - cutmix3d
- Textmodel
    - Word-level Mixup (https://arxiv.org/pdf/1905.08941.pdf)

# Models & performance
- Speechmodel 
    - AttnCNN : Efficientnet Backbone + AttnBiLSTM(with FANModule(https://arxiv.org/pdf/1907.00193.pdf)) 
- Facemodel
    - CLSTM : Almost same as Speechmodel, but image does not contain time-attribute itself while speech does. Extract time-features with multi-forwarding of backbone-model
- Textmodel
    - Vanilla-Transformer-Encoder(with small version)

# Requirements
- librosa
- moviepy
- pytorch
- opencv
- numpy
- scipy
- facenet_pytorch
- tqdm
- albumentations
- apex/amp
- efficientnet_pytorch
- https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

# TODO

# Have Tried
- r2plus1d for face model -> too slow / low performance
- Naive-Attentioned-LSTM (not FAN attention) for text model -> Bad! 