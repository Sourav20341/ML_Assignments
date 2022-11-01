import math
import random
import numpy as np


class Dataset:
    size = 10000

    def init(self, size = 10000):
        self.size = size

    def get(self, noise):
        data = []
        if noise:
            noise_arr = np.random.normal(0, 0.1, [self.size, 2])
            noise_arr = np.insert(noise_arr, 2, 0, axis=1)
            for i in range(int(self.size / 2)):
                point = []
                pie = random.uniform(0, 2 * math.pi)
                point.append(math.cos(pie))
                point.append(math.sin(pie))
                point.append(0)
                data.append(point)
            for i in range(int(self.size / 2)):
                point = []
                pie = random.uniform(0, 2 * math.pi)
                point.append(math.cos(pie))
                point.append(math.sin(pie) + 3)
                point.append(1)
                data.append(point)
            data = np.array(data)
            data = np.add(data, noise_arr)
        else:
            for i in range(int(self.size / 2)):
                point = []
                pie = random.uniform(0, 2 * math.pi)
                point.append(math.cos(pie))
                point.append(math.sin(pie))
                point.append(0)
                data.append(point)
            for i in range(int(self.size / 2)):
                point = []
                pie = random.uniform(0, 2 * math.pi)
                point.append(math.cos(pie))
                point.append(math.sin(pie) + 3)
                point.append(1)
                data.append(point)
            data = np.reshape(data,(10000,3))
        np.random.shuffle(data)
        return data

def sgn(a):
    if (a < 0):
        return -1
    else:
        return 1


def PTA(X, Y_p):
    Y_size = Y_p.shape
    Y = Y_p.copy()
    for i in range(Y_size[0]):
        if Y[i] == 0:
            Y[i] = -1
    w1 = 0
    w2 = 0
    bias = 0
    while True:
        prev_wt1 = w1
        prev_wt2 = w2
        prev_bias = bias
        for i in range(Y_size[0]):
            y_pred = sgn(w1 * X[i][0] + w2 * X[i][1] + bias)
            error = Y[i] - y_pred
            if error != 0:
                w1 = w1 + error * X[i][0]
                w2 = w2 + error * X[i][1]
                bias = bias + error
        if w1 == prev_wt1 and w2 == prev_wt2 and prev_bias == bias:
            return w1, w2, bias


def PTA_with_constant_bias(X, Y_p, iter=1000):
    Y_size = Y_p.shape
    Y = Y_p.copy()
    for i in range(Y_size[0]):
        if Y[i] == 0:
            Y[i] = -1
    w1 = 0
    w2 = 0
    for _ in range(iter):
        prev_wt1 = w1
        prev_wt2 = w2
        for i in range(Y_size[0]):
            y_pred = w1 * X[i, 0] + w2 * X[i, 1]
            error = Y[i] - sgn(y_pred)
            if error != 0:
                w1 = w1 + error * X[i, 0]
                w2 = w2 + error * X[i, 1]
        if w1 == prev_wt1 and w2 == prev_wt2:
            return w1, w2

    return w1, w2

def XOR_Data():
  arr = []
  for i in range(2):
    for j in range(2):
      arr.append([i,j])
  nparray = np.array(arr)
  nparray = np.insert(nparray,2,0,axis = 1)
  for i in range(4):
      nparray[i][2] = nparray[i][0] ^ nparray[i][1]
  return nparray

def AND_Data():
  arr = []
  for i in range(2):
    for j in range(2):
      arr.append([i,j])
  nparray = np.array(arr)
  nparray = np.insert(nparray,2,0,axis = 1)
  for i in range(4):
      nparray[i][2] = nparray[i][0] and nparray[i][1]
  return nparray

def OR_Data():
  arr = []
  for i in range(2):
    for j in range(2):
      arr.append([i,j])
  nparray = np.array(arr)
  nparray = np.insert(nparray,2,0,axis = 1)
  for i in range(4):
      nparray[i][2] = nparray[i][0] or nparray[i][1]
  return nparray