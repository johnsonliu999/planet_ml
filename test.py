from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from app import net

im = Image.open("../img/20170707_181137_100b_3B_Visual/20170707_181137_100b_3B_Visual-0.png")

imarray = np.array(im)

imdata = imarray[2000:2000+20, 3000: 3000+20, 0:3]

print(net.classify(imdata.reshape((1200, 1))))
