import json
import random
import numpy as np


f = open(r'../planesnet.json')
planesnet = json.load(f)
f.close()

ims = [np.reshape(im, (1200, 1)) for im in planesnet['data']]
labels = planesnet['labels']
data = list(zip(ims, labels))
random.shuffle(data)

cut = int(len(data)*0.8)
training_data = data[:cut]
test_data = data[cut:]
