import helper, lime_model
import alexnet, inception_v3
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # dog = Image.open("data/dog.jpg")
    # automotive = Image.open("data/automotive.jpg")
    # strawberry = Image.open("data/strawberries.jpg")
    # winebottle = Image.open("data/winebottle_2.jpg")
    #
    # alex = alexnet.Alexnet()
    # alex.classify_image_alexnet(winebottle)

    img = helper.get_image_rgb('data/dogs.png')
    # plt.imshow(img)
    # plt.show()
    #
    inception = inception_v3.InceptionVThree()
    # print(inception.get_top_n_predictions(img, 5))
    #
    # alex = alexnet.Alexnet()
    # alex.classify_image_alexnet(img)
    lime = lime_model.XaiLime(inception, [img])
    print(lime.test_function())




main()
