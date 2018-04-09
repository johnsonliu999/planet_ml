from ml import Network
from data_loader import training_data, test_data

net = Network([1200, 50, 1], act_fn="sigmoid")
# net.train(training_data, 5.0)

# net = Network()
# net.load("./one_pass_5.0")
net.SGD(training_data, 30, 10, 5.0)
# net.train(training_data, 3.0, test_data=test_data)
net.save("./1200_50_1_sigmoid_SGD_30_10_5.0")