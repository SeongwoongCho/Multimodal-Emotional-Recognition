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
- RTX6000 * 2 supported by NIPA

# WORK FLOW
1. Preprocess -> CropFace from video, get normalized log-mel spectrogram from trimmed Speech
2. Training
    1. training speech-model : efficientnet
    2. (optional)training face-model : 학습을 할 경우 3d-resnet을 사용, 아닐경우 프레임별 vgg-face embedding을 사용한다.
    3. finetuning multi-modal model with pretrained weight 1,2 and text data
    
3. something considerable
    - augmentation : mixup, (speech)specaug, 
    - attentioned-pooling : multi-modal case의 경우 aligned pooling을 수행..!
    - multitask-learning vs transfer-learning
        
# CMD
1. python3 preprocess.py        
2. python train.py --mode speech --exp_name baseline --learning_rate 4e-3 --batch_size 256 --n_epoch 100 --optim adamw --weight_decay 1e-5 --num_workers 16 --coeff 0 --mixup_prob 1 --mixup_alpha 1 --label_smoothing 0.1 --amp False >> logs.txt
3. python3 train.py --mode multimodal --pretrained_speech --pretrained_face --freeze_head

## requirements

liborsa
moviepy
pytorch
opencv
numpy
scipy

facenet_pytorch
tqdm
albumentations