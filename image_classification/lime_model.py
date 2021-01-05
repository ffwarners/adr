#https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
#https://pytorch.org/docs/stable/torchvision/models.html

import matplotlib.pyplot as plt
import helper, inception_v3
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image

img = helper.get_image_rgb('data/automotive.jpg')
plt.imshow(img)
plt.show()

inception = inception_v3.InceptionVThree()
print(inception.get_top_n_predictions(img, 5))
pill_transformed = helper.get_pil_transform()

test_pred = inception.batch_predict([pill_transformed(img)])
test_pred.squeeze().argmax()

inception = inception_v3.InceptionVThree()

img = helper.get_image_rgb('data/automotive.jpg')

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(np.array(pill_transformed(img)),
                                         inception.batch_predict, # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.show()