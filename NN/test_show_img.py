"""
test_show_img.py :
the tests for displaying the images
"""

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ml import Network
from PIL import ImageFilter, ImageStat

########### test for show
im = Image.open("../img/20170707_181137_100b_3B_Visual_cropped.png")

imarray = np.asarray(im)
part = imarray[182:202, 475:495, 0:3]
print(part.shape)

part_im = Image.fromarray(part)
# part_im = part_im.convert('L').convert('1')
# part_im = part_im.filter(ImageFilter.SMOOTH)
# print(np.asarray(part_im).shape)
# plt.imshow(part_im)


plt.figure(1)

plt.subplot(151)
plt.imshow(part_im)

plt.subplot(152)
part_im = part_im.convert('L')
plt.imshow(part_im)

plt.subplot(153)
part_im = part_im.filter(ImageFilter.SMOOTH)
plt.imshow(part_im)

plt.subplot(154)
part_im = part_im.filter(ImageFilter.Kernel((3, 3), (
    1, -1, -1,
    -1, 1, -1,
    -1, -1, 1,
), 9))
plt.imshow(part_im)

plt.subplot(155)
part_im = part_im.filter(ImageFilter.CONTOUR).convert('1')
plt.imshow(part_im)

# plt.subplot(144)
# plt.imshow(part_im.filter(ImageFilter.SMOOTH_MORE).convert('1'))

plt.show()

# part = part.reshape((400, 3)).T.reshape((1200, 1))
#
# part = part.reshape((3, 400)).T.reshape(20, 20, 3)
#
# plt.imshow(part)
# plt.show()
