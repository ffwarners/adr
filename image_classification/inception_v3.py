#https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

import helper, json, torch
from torchvision import models
import torch.nn.functional as nn_functional


class InceptionVThree:
    def __init__(self):
        self.model = models.inception_v3(pretrained=True)
        self.idx2label = []
        self.cls2label = {}
        self.cls2idx = {}
        self.class_idx = None
        self.reader('data/imagenet_class_index.json')

    def get_idx2label(self):
        return self.idx2label

    def get_cls2label(self):
        return self.cls2label

    def get_cls2idx(self):
        return self.cls2idx

    def get_class_idx(self):
        return self.class_idx

    def reader(self, file):
        #Load label texts for ImageNet predictions so we know what model is predicting
        with open(file, 'r') as read_file:
            self.class_idx = json.load(read_file)
            self.idx2label = [self.class_idx[str(k)][1] for k in range(len(self.class_idx))]
            self.cls2label = {self.class_idx[str(k)][0]: self.class_idx[str(k)][1] for k in range(len(self.class_idx))}
            self.cls2idx = {self.class_idx[str(k)][0]: k for k in range(len(self.class_idx))}

    def batch_predict(self, images):
        # Now we are ready to define classification function that Lime needs.
        # The input to this function is numpy array of images where each image is ndarray of shape
        # (channel, height, width).
        # The output is numpy array of shape (image index, classes) where each value in array should be probability for
        # that image, class combination.
        self.model.eval()
        preprocess_transform = helper.get_preprocess_transform()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)
        logits = self.model(batch)
        probs = nn_functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def predict_image(self, img):
        img_t = helper.get_input_tensors(img)
        self.model.eval()
        logits = self.model(img_t)
        return logits

    def get_top_n_predictions(self, img, n):
        logits =  self.predict_image(img)
        probs = nn_functional.softmax(logits, dim=1)
        probs_top_n = probs.topk(n)
        return tuple((p,c, self.idx2label[c]) for p, c in zip(probs_top_n[0][0].detach()
                                                       .numpy(), probs_top_n[1][0].detach().numpy()))
