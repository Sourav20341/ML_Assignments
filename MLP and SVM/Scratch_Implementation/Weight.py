import numpy as np


class zero_init:

    @staticmethod
    def func(m, n):
        return np.zeros((m, n))


class random_init:

    @staticmethod
    def func(m, n):
        return np.random.rand(m,n) * 0.001


class normal_init:

    @staticmethod
    def func(m, n):
        return np.random.normal(size=(m,n)) * 0.1
