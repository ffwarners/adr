#https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

import torch
from torchvision import models
from torchvision import transforms

def classify_image_alexnet(img):
 alexnet = models.alexnet(pretrained=True)
 transform = transforms.Compose([  # [1]
  transforms.Resize(256),  # [2]
  transforms.CenterCrop(224),  # [3]
  transforms.ToTensor(),  # [4]
  transforms.Normalize(  # [5]
   mean=[0.485, 0.456, 0.406],  # [6]
   std=[0.229, 0.224, 0.225]  # [7]
  )])
 img_t = transform(img)
 batch_t = torch.unsqueeze(img_t, 0)

 alexnet.eval()

 out = alexnet(batch_t)

 with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

 _, index = torch.max(out, 1)

 percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

 print(labels[index[0]], percentage[index[0]].item())
