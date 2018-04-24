"""
sliding_window :
The sliding window method to scan through the satellite image and mark out the planes
"""
from PIL import Image, ImageFilter
import numpy as np
from matplotlib import pyplot as plt, patches
from plane_net import Net

net = Net()
net.load("./models/gray_400_50_2_sigmoid_3.0_30")

im = Image.open("../img/20170707_181137_100b_3B_Visual_cropped.png")

fig, ax = plt.subplots(1)
ax.imshow(im)

def sliding_window(image, step_size, net):
    ret = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if x + 20 > image.shape[0] or y + 20 > image.shape[1]: continue
            label = net.classify(preprocess(image[x:x + 20, y:y + 20, 0:3].reshape((400, 3)).T.reshape((1200, 1))))
            if label == 1:
                ret.append((y, x))
        print("finished %d line" % y)
    return ret


# ndarray with shape 1200 * 1
def preprocess(data):
    im = data.reshape((3, 400)).T.reshape((20, 20, 3))
    # im = Image.fromarray(im).filter(ImageFilter.SMOOTH).convert('L').convert('1')
    im = Image.fromarray(im).convert('L')
    return np.asarray(im).reshape(400, 1)
    # return data


cors = sliding_window(np.array(im), 5, net)
print(cors)
for cor in cors:
    rect = patches.Rectangle(cor, 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
