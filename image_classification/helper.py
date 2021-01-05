import os
from torchvision import transforms
from PIL import Image


def get_image_rgb(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_preprocess_transform():
    #take resized, cropped image and apply whitening
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transform


def get_pil_transform():
    #take PIL image, resize and crop it
    transformed = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    return transformed


def read_labels(file):
    with open(file) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


#W e need to convert this image to Pytorch tensor and also apply whitening
# as used by our pretrained model
# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


def read_labels(file):
    with open(file) as f:
        labels = [line.strip() for line in f.readlines()]
        return labels