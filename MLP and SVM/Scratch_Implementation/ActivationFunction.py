import numpy as np


class sigmoid:

    @staticmethod
    def func(x):
        return 1 / (1 - np.exp(-x))

    @staticmethod
    def derivative(x):
        return sigmoid.func(x) * (1 - sigmoid.func(x))


class tanh:

    @staticmethod
    def func(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x) ** 2


class ReLU:

    @staticmethod
    def func(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        x[x<0] = 0
        x[x>0] = 1
        return x


class Leaky_ReLU:

    @staticmethod
    def func(x, a):
        r = np.maximum(0, x)
        if r <= 0:
            return a * x

        else:
            return x

    @staticmethod
    def derivative(x, a):
        if Leaky_ReLU.func(x, a) == x:
            return 1
        else:
            return a


class softmax:

    @staticmethod
    def func(x):
        return np.exp(x) / np.sum(np.exp(x))


class Linear:

    @staticmethod
    def func(x):
        return x

    @staticmethod
    def derivative(self):
        return 1
