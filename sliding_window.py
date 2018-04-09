from PIL import Image
import numpy as np
from matplotlib import pyplot as plt, patches
from ml import Network


net = Network()
net.load("./models/one_pass_5.0")
print(net.weights[0].shape)

im = Image.open("../img/20170707_181137_100b_3B_Visual_cropped.png")

fig, ax = plt.subplots(1)
ax.imshow(im)
#
# rect = patches.Rectangle((2000, 3000), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
#
# ax.add_patch(rect)
#
# plt.imshow(im)
# plt.show()


# imgplot = imgplot.addsubplot()

# imarray = np.array(im)

# imdata = imarray[2000:2000+20, 3000: 3000+20, 0:3]

# print(net.classify(imdata.reshape((1200, 1))))

def sliding_window(image, step_size, net):
    ret = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if x + 20 > image.shape[0] or y + 20 > image.shape[1]: continue
            label = net.classify(image[x:x + 20, y:y + 20, 0:3].reshape((400, 3)).T.reshape((1200, 1)))
            if label == 1:
                ret.append((y, x))
        print("finished %d line" % y)
    return ret


cors = sliding_window(np.array(im), 2, net)
print(cors)
for cor in cors:
    rect = patches.Rectangle(cor, 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
