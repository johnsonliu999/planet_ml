"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import codecs
import random
import json
# Third-party libraries
from pprint import pprint

import numpy as np


class Network(object):
    def __init__(self, sizes=None, act_fn=None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        if sizes:
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
            self.act_fn = act_fn
            self.activate = globals()[act_fn]
            self.activate_prime = globals()["%s_prime" % act_fn]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activate(np.dot(w, a) + b)
        return a

    # one pass
    def train(self, training_data, eta, test_data=None):
        self.update_mini_batch(training_data, eta)
        if test_data:
            n_test = len(test_data)
            print("One pass : {0} / {1}".format(self.evaluate(test_data), n_test))

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # print("test_data 0" , test_data[0])
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # print("x shape", x.shape)
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            # print("z shape 1: ", z.shape)
            # print("w shape", w.shape)
            # print("activation shape", activation.shape)
            zs.append(z)
            activation = self.activate(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.activate_prime(zs[-1])

        # print("delta shape", delta.shape)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # print("layers: ", self.num_layers)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activate_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # print("z shape: ", z.shape)
            # print("sp shape: ", sp.shape)
            # print("shape: ", delta.shape)
            # print("type :", type(delta))
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # test_results = [(np.argmax(self.feedforward(x)), y)
        #                 for (x, y) in test_data]
        test_results = [(round(self.feedforward(x)[0][0]), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    # data : img ndarray with shape (1200, 1)
    def classify(self, data):
        return int(round(self.feedforward(data)[0][0]))

    def save(self, path):
        data = {'sizes': self.sizes,
                'num_layers': self.num_layers,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'act_fn': self.act_fn}
        json.dump(data, codecs.open(path, 'w', encoding='utf-8'))

    def load(self, path):
        data = json.loads(codecs.open(path, 'r', encoding='utf-8').read())
        print(data.keys())
        self.num_layers = data["num_layers"]
        self.sizes = data["sizes"]
        self.biases = [np.array(b) for b in data["biases"]]
        self.weights = [np.array(w) for w in data["weights"]]
        self.act_fn = data["act_fn"]
        self.activate = globals()[self.act_fn]
        self.activate_prime = globals()["%s_prime" % self.act_fn]


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def identity(z):
    return z.copy()


def identity_prime(z):
    return 1


def relu(z):
    return np.where(z < 0, 0., z)


def relu_prime(z):
    return np.where(z < 0, 0., 1.)
