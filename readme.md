# WORK FLOW
0. keyword
    - multitask learning
1. MTCNN으로 얼굴 부분 인식 -> 프리프로세스, 음성도 프리프로세스
2. 학습
    - model 1 = face recognition model
    - model 2 = speech recognition model
    - 각각을 전부 attentioned-pooling으로 처리
    - feature_face, feature_speech, feature_text를 dense해서 multi modal output 만듬
    - loss_ratio l 은 스케쥴링. 1에서 점점 0으로 ? 
    
    - loss_face, loss_speech, loss_modal
    - loss = (l*(loss_face + loss_speech) + loss_modal)/(l+1)
    
3. augmentation
    - mixup
        - face, speech를 동시에 mix ... 음 text가 mix가 안될 것 같은데 각각 학습시켜야 하나. text는 finetuning stage에서 쓸수도 있고..!? 

        
# CMD

1. python3 preprocess.py
    -- 
        
2. python3 train.py --mode speech
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
