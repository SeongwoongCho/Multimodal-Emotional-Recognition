import albumentations

# sudo pip install git+https://github.com/okankop/vidaug
# from vidaug import augmentors as va

mean = [0.45]
std = [0.23]

def get_speech_transform(is_train):
    if is_train:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width=450),
                albumentations.RandomCrop(height = 128, width = 450),
                albumentations.RandomBrightnessContrast(brightness_limit = 0.2,contrast_limit = 0.2, p = 0.5),
                albumentations.Normalize(mean=mean, std=std,max_pixel_value = 1.0)
            ]
        )
    else:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width=450),
                albumentations.CenterCrop(height = 128, width = 450),
                albumentations.Normalize(mean=mean, std=std,max_pixel_value = 1.0)
            ]
        )
    return transform

def get_face_transform(is_train):
    pass