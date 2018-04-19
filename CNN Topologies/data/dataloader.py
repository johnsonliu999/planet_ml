import json
import random
import math
import numpy as np

'''
Loads the original data and splits it into train and test data sets
'''

# Load planesnet data
f = open('planesnet.json')
planesnet = json.load(f)
f.close()

# Shuffle:
dataXy = list(zip(planesnet['data'], planesnet['labels']))
random.shuffle(dataXy)
X, y = zip(*dataXy)

# Train, test data separation.
train_set = 0.9
train_records = math.floor(train_set * len(X))

Xtrain = X[:train_records]
ytrain = y[:train_records]
Xtest = X[train_records + 1:]
ytest = y[train_records + 1:]

#Dump the data onto files
json.dump({"data":Xtrain},open("Xtrain","w"))
json.dump({"data":ytrain},open("ytrain","w"))
json.dump({"data":Xtest},open("Xtest","w"))
json.dump({"data":ytest},open("ytest","w"))

print("done")
