# directory structure

```
/
├── features/
│   ├── train/
│   ├── val/ 
│   └── test/
│       ├── speech/
│       └── video/ 
├── qia2020/
│   ├── train/
│   ├── val/ 
│   └── test/
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

# WORK FLOW
1. Preprocess -> CropFace from video, get normalized log-mel spectrogram from trimmed Speech
2. Training
    1. training speech-model : efficientnet
    2. (optional)training face-model : 학습을 할 경우 3d-resnet을 사용, 아닐경우 프레임별 vgg-face embedding을 사용한다.
    3. finetuning multi-modal model with pretrained weight 1,2 and text data
    
3. something considerable
    - augmentation : mixup, (speech)specaug, 
    - attentioned-pooling : multi-modal case의 경우 aligned pooling을 수행..! attention-encoder-decoder 모델을 활용
    - multitask-learning vs transfer-learning

# How to start

```
1. python3 preprocess.py

2-1. python train.py --mode speech --exp_name vanila_effb4 --learning_rate 4e-3 --batch_size 128 --n_epoch 100 --optim adamw --warmup 3 --weight_decay 5e-5 --num_workers 16 --coeff 4 --mixup_prob 1 --mixup_alpha 1 --label_smoothing 0.1 --amp False >> logs.txt

2-2. python train.py --mode text --exp_name attn_lstm --learning_rate 4e-3 --batch_size 256 --n_epoch 100 --optim adamw --warmup 3 --weight_decay 5e-5 --num_workers 16 --label_smoothing 0.1 --amp False >> logs.txt

2-3. python train.py --mode face --exp_name attn_lstm --learning_rate 4e-3 --batch_size 256 --n_epoch 100 --optim adamw --warmup 3 --weight_decay 5e-5 --num_workers 16 --label_smoothing 0.1 --amp False >> logs.txt

3. python3 train.py --mode multimodal --exp_name 201103 --speech_load_weights './logs/speech/vanila_effb4/94_best_1.5341.pth' --text_load_weights './logs/text/attn_lstm/99_best_1.6258.pth' --face_load_weights './logs/face/attn_lstm/61_best_1.8634.pth' --learning_rate 4e-3 --batch_size 128 --n_epoch 100 --optim adamw --warmup 1 --weight_decay 1e-5 --num_workers 16 --coeff 4 --lam 0 --label_smoothing 0.1 --amp False >> logs.txt
```

## Requirements
- liborsa
- moviepy
- pytorch
- opencv
- numpy
- scipy
- facenet_pytorch
- tqdm
- albumentations

## New ideas
https://github.com/titu1994/MLSTM-FCN