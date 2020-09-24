import albumentations

# sudo pip install git+https://github.com/okankop/vidaug
# from vidaug import augmentors as va

mean = [0]
std = [1]

def get_speech_transform(is_train):
    ## border mode 0 : constant padding, border mode 4 : repeats
    if is_train:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width = 400,border_mode = 0),
                albumentations.RandomCrop(height = 128, width = 400),
                albumentations.RandomBrightnessContrast(brightness_limit = 0.3,contrast_limit = 0.3, p = 0.7),
#                albumentations.Normalize(mean=mean, std=std,max_pixel_value = 255.0)
            ]
        )
    else:
        transform = albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 128, min_width=400,border_mode = 0),
                albumentations.CenterCrop(height = 128, width = 400),
#                albumentations.Normalize(mean=mean, std=std,max_pixel_value = 255.0)
            ]
        )
    return transform

def get_face_transform(is_train):
    if is_train:
        return albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 60, min_width = 512,border_mode = 0),
                albumentations.RandomCrop(height = 60, width = 512)
            ]
        )
    else:
        return albumentations.Compose(
            [
                albumentations.PadIfNeeded(min_height = 60, min_width = 512,border_mode = 0),
                albumentations.CenterCrop(height = 60, width = 512)
            ]
        )
    
def get_text_transform(is_train):
    return albumentations.PadIfNeeded(min_height = 38, min_width = 200,border_mode = 0)