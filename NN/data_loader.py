import json
import random
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

DATA_PATH = '../CNN Topologies/data/'

def load_data():
    f = open(DATA_PATH + 'xtrain')
    xtrain = json.load(f)["data"]
    print(np.array(xtrain[0]).shape)
    f.close()

    f = open(DATA_PATH + 'ytrain')
    ytrain = json.load(f)["data"]
    f.close()

    f = open(DATA_PATH + 'xtest')
    xtest = json.load(f)["data"]
    f.close()

    f = open(DATA_PATH + 'ytest')
    ytest = json.load(f)["data"]
    f.close()

    xtrain = [grayify(np.reshape(im, (1200,1)).astype('uint8')) for im in xtrain]
    xtest = [grayify(np.reshape(im, (1200,1)).astype('uint8')) for im in xtest]
    training_data = list(zip(xtrain, ytrain))
    test_data = list(zip(xtest, ytest))

    return training_data, test_data


# return 1200 * 1 shape
def grayify(data):
    im = data.reshape((3, 400)).T.reshape((20, 20, 3))
    im = Image.fromarray(im).filter(ImageFilter.SMOOTH).convert('L').convert('1')
    # im = Image.fromarray(im).convert('L')
    return np.asarray(im).reshape(400, 1)
    # return data
