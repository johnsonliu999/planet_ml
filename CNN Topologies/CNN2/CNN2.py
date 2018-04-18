# from __future__ import division, print_function, absolute_import

import json
import numpy as np
import tflearn
import random
import os
import math

from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Supress Tensorflow warnings when launching tensorboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Data Path
Xtrain_path = "../data/Xtrain"
Xtest_path = "../data/Xtest"
ytrain_path = "../data/ytrain"
ytest_path = "../data/ytest"

#model path
model_path = "./model/cnn2.model"

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
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, _dropout)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=_learning_rate)
model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=0)

#comment train/load sections accordingly
# Train the model
model.fit(Xtrain, ytrain, n_epoch=10, shuffle=True, validation_set=.2,
          show_metric=True, batch_size=128, run_id='planesnet')

# Save the model so that we can use the trained model
model.save(model_path)

#load the trained model
#model.load(model_path)


# Evaluation:
# True Positive(tp), True Negative(tn), False Positive(fp), False Negative(fn)
tp, tn, fp, fn = 0, 0, 0, 0
for j in range(len(Xtest)):
    img = Xtest[j] / 255.
    img = img.reshape((3, 400)).T.reshape((20, 20, 3))
    label = ytest[j]

    # Predict the image class
    prediction = model.predict_label([img])[0][0]

    # class - 0 = 'no-plane', 1 = 'plane'
    if prediction == 1 and label == prediction:
        tp += 1
    elif prediction == 0 and label == prediction:
        tn += 1
    elif prediction == 1 and label == 0:
        fp += 1
    else:
        fn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
f_measure = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F-measure:", f_measure)
