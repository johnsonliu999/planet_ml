from ml import Network
from data_loader import training_data, test_data


net = Network([1200, 30, 1], act_fn="sigmoid")
# net.train(training_data, 5.0)

# net = Network()
# net.load("./one_pass_5.0")
net.SGD(training_data, 10, 5, 2.0, test_data=test_data)
# net.train(training_data, 3.0, test_data=test_data)
net.save("./models/" + "1200_30_1_sigmoid_SGD_10_5_2.0")
