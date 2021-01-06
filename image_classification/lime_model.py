#https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
#https://pytorch.org/docs/stable/torchvision/models.html

import matplotlib.pyplot as plt
import helper, inception_v3
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image
import helper, json, torch
from torchvision import models
import torch.nn.functional as nn_functional


# We are getting ready to use Lime. Lime produces the array of images from original input image by pertubation algorithm.
# So we need to provide two things:
# (1) original image as numpy array
# (2) classification function that would take array of purturbed images as input and produce the probabilities for each
# class for each image as output.
# For Pytorch, first we need to define two separate transforms:
# (1) to take PIL image, resize and crop it
# (2) take resized, cropped image and apply whitening.

class XaiLime:
    def __init__(self, model, images):
        self.model = model.model
        self.images = images

    def batch_predict(self):
        # Now we are ready to define classification function that Lime needs.
        # The input to this function is numpy array of images where each image is ndarray of shape
        # (channel, height, width).
        # The output is numpy array of shape (image index, classes) where each value in array should be probability for
        # that image, class combination.
        self.model.eval()
        preprocess_transform = helper.get_preprocess_transform()
        batch = torch.stack(tuple(preprocess_transform(i) for i in self.images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)
        logits = self.model(batch)
        probs = nn_functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def test_function(self):
        pill_transformed = helper.get_pil_transform()
        test_pred = self.batch_predict()
        return test_pred.squeeze().argmax()



# img = helper.get_image_rgb('data/dogs.jpg')
# plt.imshow(img)
# plt.show()
#
# inception = inception_v3.InceptionVThree()
# print(inception.get_top_n_predictions(img, 5))
# pill_transformed = helper.get_pil_transform()
#
# test_pred = inception.batch_predict([pill_transformed(img)])
# test_pred.squeeze().argmax()
#
#

# explainer = lime_image.LimeImageExplainer()
#
# explanation = explainer.explain_instance(np.array(pill_transformed(img)),
#                                          inception.batch_predict, # classification function
#                                          top_labels=5,
#                                          hide_color=0,
#                                          num_samples=1000) # number of images that will be sent to classification function
#
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# img_boundry1 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry1)
# plt.show()
#
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# img_boundry2 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry2)
# plt.show()