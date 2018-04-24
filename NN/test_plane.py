import random

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageFilter


######### test for training
from plane_net import Net
from data_loader import load_data


training_data, test_data = load_data()

z_acc = []
o_acc = []
acc = []

net = Net([400, 50, 2], "sigmoid", 3.0, 30)
# net = Net()
# net.load("./models/gray_400_50_2_sigmoid_3.0_30")
for i in range(50):
    random.shuffle(training_data)
    print("******* %d ******" % i)
    net.train(training_data)
    z_p, z, o_p, o = net.test(test_data)
    z_acc.append(z_p / z)
    o_acc.append(o_p / o)
    acc.append((z_p+o_p) / (z+o))
    print("zero: %f, one: %f" % (z_p/z, o_p/o))


tn, n, tp, p = net.test(test_data)
precision = tp / (tp +  p - tp)
recall = tp / (tp + n - tn)
accuracy = (tp + tn) / (n + p)
f_measure = 2 * (precision * recall) / (precision + recall)

print ("tn", tn)
print ("fn", n - tn)
print("tp", tp)
print("fp", p - tp)

print("precision : ", precision)
print("recal : ", recall)
print("accuracy : ", accuracy)
print("f_measure : ", f_measure)

# precision :  0.6708542713567839
# recal :  0.8030075187969925
# accuracy :  0.8771491090965927
# f_measure :  0.731006160164271

# net.save("./models/gray_400_50_2_sigmoid_3.0_30")

print("z_acc: ", z_acc)
print("o_acc: ", o_acc)
print("acc: ", acc)
plt.plot(z_acc, label='0s')
plt.plot(o_acc, label='1s')
plt.plot(acc, label='overall')
plt.legend()

plt.show()
print(net.activate_prime)
print(net.ws[1].shape)
