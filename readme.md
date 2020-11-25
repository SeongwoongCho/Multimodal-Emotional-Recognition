# Multimodal-Emotional-Recognition

# [대회 제출 검증용] How to reproduce leaderboard result

```
1. 라이브러리들을 설치한다.(학습이 아닌 inference를 위한 라이브러리임) 전부 pip install (library)로 설치가능합니다.
    - torch == 1.7.0
    - torchvision == 0.8.1
    - librosa
    - moviepy
    - opencv-python
    - pandas
    - ttach
    - numpy
    - facenet-pytorch
    - efficientnet_pytorch
2. test1, test2 데이터 셋을 qia2020/test/ 폴더에 옮긴다. 즉, qia2020/test/ 폴더에 .npz, .mp4 파일들이 위치해있다.
3. test3 데이터 셋은 2020-3/test3/ 폴더에 있다.즉, 2020-3/test3/ 폴더에 .npz, .mp4 파일들이 위치해있다.
4. python preprocess.py 를 실행한다
5. make_submission.py를 실행했을 때 나오는 3개의 csv파일중 test2_submission.csv, test3_submission.csv 가 각각 phase2,3의 리더보드를 복원함
6. make_submission.py의 64~71, 124-130, 174-181 번째 줄을 주석처리한후 make_submission.py를 실행한다. 이 때 나오는 3개의 csv파일중 test1_submission.csv 가 각각 phase1의 리더보드를 복원함

```

# Dataset
- QIA2020/MERC2020
    - video / word2vec embedding of sentence is given!
    - labels for each attribute(face,speech,integrate)
    - The target task is maximizing the accuracy of integrate-emotion-label(multimodal label)
    - 7 emotions! 'hap','ang','dis','fea','sad','neu','sur'
- You can train on your own dataset by 

# Directory Structure

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
├── 2020-3/
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
$ python preprocess.py

$ python train.py --mode speech --exp_name AttnCNN-B4 --learning_rate 4e-3 --batch_size 256 --n_epoch 100 \
--optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --mixup_prob 1 --mixup_alpha 1 \
--label_smoothing 0.1 --amp True >> ./logs/logs.txt

$ python train.py --mode face --exp_name CLSTM-B0-latest --learning_rate 4e-3 --batch_size 32 --n_epoch 20 \
--optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --face_coeff 0 --cutmix_prob 1 --cutmix_beta 1 \
--label_smoothing 0.1 --amp True >> ./logs/logs.txt

$ python train.py --mode text --exp_name transformer_12_8 --learning_rate 3e-4 --batch_size 128 \
--n_epoch 100 --optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 \
--label_smoothing 0.1 --amp False >> ./logs/logs.txt



(!Amp does not work in the Transformer)

## seed 42
$ python train.py --mode multimodal --exp_name mixup-lam_2-latest \
--speech_load_weights ./logs/speech/AttnCNN-B4/50_best_1.5379.pth \
--text_load_weights ./logs/text/transformer_12_8/33_best_1.5556.pth \
--face_load_weights ./logs/face/CLSTM-B0-latest/10_best_1.6658.pth \
--learning_rate 3e-4 --batch_size 16 --n_epoch 20 --optim sgd --warmup 0 \
--weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --face_coeff 0 \
--mixup_prob 1 --mixup_alpha 1 --lam 2 --label_smoothing 0.1 --amp False \
>> ./logs/logs.txt

## seed 4242
$ python train.py --mode multimodal --exp_name mixup-lam_2-latest-newseed \
--speech_load_weights ./logs/speech/AttnCNN-B4-newseed/52_best_1.5434.pth \
--text_load_weights ./logs/text/transformer_12_8-newseed/27_best_1.5660.pth \
--face_load_weights ./logs/face/CLSTM-B0-latest-newseed/14_best_1.6082.pth \
--learning_rate 3e-4 --batch_size 16 --n_epoch 20 --optim sgd --warmup 0 \
--weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --face_coeff 0 \
--mixup_prob 1 --mixup_alpha 1 --lam 2 --label_smoothing 0.1 --amp False \
>> ./logs/logs.txt

## seed 424242
$ python train.py --mode multimodal --exp_name mixup-lam_2-latest-newnewseed \
--speech_load_weights ./logs/speech/AttnCNN-B4-newnewseed/41_best_1.5449.pth \
--text_load_weights ./logs/text/transformer_12_8-newnewseed/23_best_1.5660.pth \
--face_load_weights ./logs/face/CLSTM-B0-latest-newnewseed/10_best_1.6133.pth \
--learning_rate 3e-4 --batch_size 16 --n_epoch 20 --optim sgd --warmup 0 \
--weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --face_coeff 0 \
--mixup_prob 1 --mixup_alpha 1 --lam 2 --label_smoothing 0.1 --amp False \
>> ./logs/logs.txt

## seed 1234
python train.py --mode multimodal --exp_name mixup-lam_2-seed1234-bigger-models \
--speech_load_weights ./logs/speech/AttnCNN-B7/52_best_1.5377.pth \
--text_load_weights ./logs/text/transformer_12_8_seed1234/23_best_1.5553.pth \
--face_load_weights ./logs/face/CLSTM-B1-frame16/13_best_1.5900.pth \
--learning_rate 6e-4 --batch_size 8 --n_epoch 20 --optim sgd --warmup 0 \
--weight_decay 5e-5 --num_workers 16 --speech_coeff 7 --face_coeff 1 \
--mixup_prob 1 --mixup_alpha 1 --lam 2 --label_smoothing 0.1 --amp False \
>> ./logs/logs.txt

python train.py --mode speech --exp_name AttnCNN-B4-newnewseed --learning_rate 4e-3 --batch_size 256 --n_epoch 100 \
--optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --mixup_prob 1 --mixup_alpha 1 \
--label_smoothing 0.1 --amp True >> ./logs/logs.txt && python train.py --mode face --exp_name CLSTM-B0-latest-newnewseed --learning_rate 4e-3 --batch_size 32 --n_epoch 20 \
--optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 --face_coeff 0 --cutmix_prob 1 --cutmix_beta 1 \
--label_smoothing 0.1 --amp True >> ./logs/logs.txt && python train.py --mode text --exp_name transformer_12_8-newnewseed --learning_rate 3e-4 --batch_size 128 \
--n_epoch 100 --optim ranger --warmup 0 --weight_decay 5e-5 --num_workers 16 \
--label_smoothing 0.1 --amp False >> ./logs/logs.txt && python train.py --mode multimodal --exp_name mixup-lam_2-latest-newnewseed \
--speech_load_weights ./logs/speech/AttnCNN-B4-newseed/52_best_1.5434.pth \
--text_load_weights ./logs/text/transformer_12_8-newseed/27_best_1.5660.pth \
--face_load_weights ./logs/face/CLSTM-B0-latest-newseed/14_best_1.6082.pth \
--learning_rate 3e-4 --batch_size 16 --n_epoch 20 --optim sgd --warmup 0 \
--weight_decay 5e-5 --num_workers 16 --speech_coeff 4 --face_coeff 0 \
--mixup_prob 1 --mixup_alpha 1 --lam 2 --label_smoothing 0.1 --amp False \
>> ./logs/logs.txt

```

# Preprocessing
- The preprocessing is optimized for the "QIA(Qualcomm innovation awards)2020/MERC(multimodal-emotional-recognition-challenge)2020" dataset.

- Speech -> denoising / Remove Silence / Log-scale Mel spectrogram
- Video -> Frame-level FaceCrop(MTCNN)/Resize
- Text -> word2vec embedding of the each word of a sentence (This is given on the QIA2020/MERC2020 challenge!)

# Augmentation
- Speechmodel
    - RandomClip
    - RandomBrightnessContrast
    - SpecAug
    - Mixup
- Facemodel
    - RandomFrameSelection
    - RandomHorizontalFlip
    - cutmix3d
- Textmodel
    - Word level mixup!

# Models & performance
- Speechmodel 
    - AttnCNN : Efficientnet Backbone + BiLSTM w/ FANModule(https://arxiv.org/pdf/1907.00193.pdf) 
    - best valid loss : 1.537871
    - best valid accuracy : 43.2812%
- Facemodel
    - CLSTM : Almost same as Speechmodel, but image does not contain time-attribute itself while speech does. Extract time-features with multi-forwarding of backbone-model
    - USE pretrained - Efficientnet b0 as backbone!
    - best valid loss :
    - best valid accuracy : 
- Textmodel
    - Vanilla-Transformer-Encoder(8head, 12layer)
    - best valid loss : 1.559641
    - best valid accuracy : 43.4048%
- multimodal
    - no scheduling!
    - best valid loss :
    - best valid accuracy : 
    - MERC2020 test1/test2/test3 accuracy : 

# Inference!
- 12 TTA on FaceImage! (Horizontal Flip, Vertical Flip, Multiply 0.9,1.0,1.1) 
- Sharpening T = 0.1 (exactly, not sharpening but smoothing)
- seed = 42, seed = 4242 - 2 model ensemble

# Requirements
- python == 3.6
- pytorch == 1.5.1
- librosa
- moviepy
- opencv
- numpy
- scipy
- facenet_pytorch
- tqdm
- albumentations
- apex/amp
- efficientnet_pytorch
- https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
- https://github.com/qubvel/ttach

# Have Tried
- r2plus1d for face model -> too slow training / convergence.
- Naive-Attentioned-LSTM (not FAN attention) for text model -> Bad! 
- AutoAugment(adjust different AutoAugment for the different video frame) for face model -> does not work!
- Pretrained-Tunit Content encoder -> does not work well! 

# TODO 
- word mixup! 
- VGGISH for audio embedding