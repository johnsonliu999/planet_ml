from __future__ import division, print_function, absolute_import

import json
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import random
from PIL import Image
from matplotlib import pyplot as plt, patches

#Data Path
Xtrain_path = "../data/Xtrain"
Xtest_path = "../data/Xtest"
ytrain_path = "../data/ytrain"
ytest_path = "../data/ytest"

#model path
model_path = "./model/VGGnet.model"
images_path = "../images/"

#load train data - features
Xtrain = json.load(open(Xtrain_path))['data']
Xtrain = np.array(Xtrain) / 255.
Xtrain = Xtrain.reshape([-1, 3, 20, 20]).transpose([0, 2, 3, 1])

#load train data - labels
ytrain = json.load(open(ytrain_path))['data']
ytrain = np.array(ytrain)
ytrain = to_categorical(ytrain, 2)

#load test data - features
Xtest = json.load(open(Xtest_path))['data']
Xtest = np.array(Xtest)

#load test data - labels
ytest = json.load(open(ytest_path))['data']
ytest = np.array(ytest)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
# Hyper params:
_learning_rate = 0.001
_dropout = 0.5

#sliding_window_size
sliding_window_size = 4

# Building 'VGG Network'
network = input_data(shape=[None, 20, 20, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, _dropout)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, _dropout)
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=_learning_rate)

#comment train/load sections accordingly
# Train the model
model = tflearn.DNN(network, tensorboard_dir='log',tensorboard_verbose=0)

#load the trained model
model.load(model_path)

img = Image.open(images_path+"image1.jpg")

fig, ax = plt.subplots(1)
ax.imshow(img)

#Move the 20*20 window across the image
def sliding_window(image, step_size, model):
    ret = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if x + 20 > image.shape[0] or y + 20 > image.shape[1]: continue
            img = image[x:x + 20, y:y + 20, 0:3].reshape((400, 3)).T.reshape((1200, 1))/ 255.
            img = img.reshape((3, 400)).T.reshape((20, 20, 3))
            label = model.predict_label([img])[0][0]
            if label == 1:
                ret.append((y, x))
        print("finished %d line" % y)
    return ret


coordinates = sliding_window(np.array(img), sliding_window_size, model)
print(coordinates)

#Draw the rectangle using the co-ordinates
for pos in coordinates:
    rect = patches.Rectangle(pos, 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

#Display the plot
plt.show()
