import random

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ml import Network
from PIL import ImageFilter


from plane_net import Net
from data_loader import load_data
import mnist_loader

training_data, test_data = load_data()

net = Net()
net.load("./models/400_100_2_sigmoid_2.0_20")

net.test(test_data)
