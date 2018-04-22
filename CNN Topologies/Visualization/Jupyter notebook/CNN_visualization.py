from __future__ import division, print_function, absolute_import

import json,random,math,os
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Data Path
Xtrain_path = "../../data/Xtrain"
Xtest_path = "../../data/Xtest"
ytrain_path = "../../data/ytrain"
ytest_path = "../../data/ytest"

#model path
model_path = "E:/NEU/FAI/Project/FAI_Project_Git/CNN Topologies/CNN2/model/cnn2.model"

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

network = input_data(shape=[None, 20, 20, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
conv1 = conv_2d(network, 32, 3, activation='relu')
pool1 = max_pool_2d(conv1, 2)
conv2 = conv_2d(pool1, 64, 3, activation='relu')
conv3 = conv_2d(conv2, 64, 3, activation='relu')
pool2 = max_pool_2d(conv3, 2)
fc1 = fully_connected(pool2, 512, activation='relu')
dp1 = dropout(fc1, _dropout)
fc2 = fully_connected(dp1, 2, activation='softmax')
network = regression(fc2, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=_learning_rate)
model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=0)


#comment train/load sections accordingly
# Train the model
# model.fit(Xtrain, ytrain, n_epoch=10, shuffle=True, validation_set=.2, show_metric=True, batch_size=128, run_id='planesnet')
#
# # Save the model so that we can use the trained model
# model.save(model_path)

#load the trained model
model.load(model_path)

import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

m_conv1 = tflearn.DNN(conv1, session=model.session)
m_conv2 = tflearn.DNN(conv2, session=model.session)
m_conv3 = tflearn.DNN(conv3, session=model.session)

for j in range(len(Xtest)):
    img = Xtest[j] / 255.
    img = img.reshape((3, 400)).T.reshape((20, 20, 3))
    label = ytest[j]

    # Display the image
    #     plt.imshow(img)
    #     plt.show()

    # Predict the image class
    prediction = model.predict_label([img])[0][0]

    #     fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    if label ==1 :
        o_conv1 = m_conv1.predict([img])
        for i in range(32):
            plt.subplot(8, 4, i + 1)
            plt.imshow(o_conv1[0][:, :, i])
        plt.show()

        o_conv2 = m_conv2.predict([img])
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(o_conv2[0][:, :, i])
        plt.show()

        o_conv3 = m_conv3.predict([img])
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(o_conv3[0][:, :, i])

        plt.show()

        # Output acutal and predicted class - 0 = 'no-plane', 1 = 'plane'
        print('Actual Class: ' + str(label))
        print('Predicted Class: ' + str(prediction))