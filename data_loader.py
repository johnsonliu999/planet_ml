import json
import random
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt


def load_data():
    f = open(r'../planesnet.json')
    planesnet = json.load(f)
    f.close()

    ims = [grayify(np.reshape(im, (1200, 1)).astype('uint8')) for im in planesnet['data']]
    labels = planesnet['labels']
    data = list(zip(ims, labels))
    random.shuffle(data)

    cut = int(len(data) * 0.6)
    training_data = data[:cut]
    test_data = data[cut:]
    return training_data, test_data


# return 1200 * 1 shape
def grayify(data):
    im = data.reshape((3, 400)).T.reshape((20, 20, 3))
    # im = Image.fromarray(im).filter(ImageFilter.SMOOTH).convert('L').convert('1')
    im = Image.fromarray(im).convert('L')
    return np.asarray(im).reshape(400, 1)
    # return data
