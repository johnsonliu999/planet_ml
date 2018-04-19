import random

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ml import Network
from PIL import ImageFilter

############ test for show
# im = Image.open("../img/20170707_181137_100b_3B_Visual_cropped.png")
#
# imarray = np.asarray(im)
# part = imarray[182:202, 475:495, 0:3]
# print(part.shape)
#
# part_im = Image.fromarray(part)
# # part_im = part_im.convert('1')
# part_im = part_im.filter(ImageFilter.SMOOTH)
# print(np.asarray(part_im).shape)
# plt.imshow(part_im)
# plt.show()
#
# # part = part.reshape((400, 3)).T.reshape((1200, 1))
# #
# # part = part.reshape((3, 400)).T.reshape(20, 20, 3)
# #
# # plt.imshow(part)
# # plt.show()


######### test for training
from plane_net import Net
from data_loader import load_data

training_data, test_data = load_data()

z_acc = []
o_acc = []
acc = []

net = Net([400, 50, 2], "sigmoid", 3.0, 30)
for i in range(100):
    random.shuffle(training_data)
    print("******* %d ******" % i)
    net.train(training_data)
    z_p, z, o_p, o = net.test(test_data)
    z_acc.append(z_p / z)
    o_acc.append(o_p / o)
    acc.append((z_p+o_p) / (z+o))
    print("zero: %f, one: %f" % (z_p/z, o_p/o))

net.save("./models/gray_400_50_2_sigmoid_3.0_30")

print("z_acc: ", z_acc)
print("o_acc: ", o_acc)
print("acc: ", acc)
plt.plot(z_acc, label='0s')
plt.plot(o_acc, label='1s')
plt.plot(acc, label='overall')
plt.legend()
plt.title("Accuracy BW")

plt.show()
print(net.activate_prime)
print(net.ws[1].shape)
