from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ml import Network

im = Image.open("../img/20170707_181137_100b_3B_Visual_cropped.png")
imarray = np.array(im)
part = imarray[182:202, 475:495, 0:3]

print(imarray[182, 475])
# plt.imshow(part)
# plt.show()

net = Network()
net.load("./models/"+"1200_30_1_sigmoid_SGD_10_5_2.0")

part = part.reshape((400, 3)).T.reshape((1200, 1))
#
# part = part.reshape((3, 400)).T.reshape(20, 20, 3)
#
# plt.imshow(part)
# plt.show()

print(net.classify(part))


