#https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

import torch, helper
from torchvision import models


class Alexnet:
    def __init__(self):
        self.labels = helper.read_labels('data/imagenet_classes.txt')
        self.model = models.alexnet(pretrained=True)

    @classmethod
    def info(cls):
        return cls.name

    def classify_image_alexnet(self, img):
        transform = helper.get_input_transform()
        img_transformed = transform(img)
        batch_t = torch.unsqueeze(img_transformed, 0)

        self.model.eval()

        out = self.model(batch_t)

        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        print(self.labels[index[0]], percentage[index[0]].item())







