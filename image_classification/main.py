from PIL import Image
import alexnet

dog = Image.open("data/dog.jpg")
automotive = Image.open("data/automotive.jpg")
strawberry = Image.open("data/strawberries.jpg")
winebottle = Image.open("data/winebottle_2.jpg")

alex = alexnet.Alexnet()
alex.classify_image_alexnet(winebottle)