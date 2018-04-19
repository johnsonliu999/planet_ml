from digit_net import Net
from mnist_loader import load_data_wrapper
from matplotlib import pyplot as plt


training_data, validation_data, test_data = load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net = Net([784, 30, 10], "sigmoid", 3.0, 20)

acc = []
for i in range(50):
    net.train(training_data)
    acc.append(net.test(test_data))

net.save("./digit_model/784_30_10_relu_3.0_20")
plt.plot(acc, label="overall")
plt.show()
