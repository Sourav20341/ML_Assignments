import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Q1
d = utils.Dataset()
Noise_DataSet = np.array(d.get(True))
Without_Noise_DataSet = np.array(d.get(False))

# Q2
sns.set(rc={'figure.figsize': (5, 8)})
sns.scatterplot(data=Noise_DataSet, hue=Noise_DataSet[:, 2], x=Noise_DataSet[:, 0], y=Noise_DataSet[:, 1])
plt.show()

sns.set(rc={'figure.figsize': (5, 8)})
sns.scatterplot(data=Without_Noise_DataSet, hue=Without_Noise_DataSet[:, 2], x=Without_Noise_DataSet[:, 0],
                y=Without_Noise_DataSet[:, 1])
plt.show()

# Q3

w1_with_Noise, w2_with_Noise, bias_with_Noise = utils.PTA(Noise_DataSet[:, :-1], Noise_DataSet[:, -1])
x = np.linspace(-2, 2)
a = -w1_with_Noise / w2_with_Noise
y2 = a * x - bias_with_Noise / w2_with_Noise
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=Noise_DataSet, hue=Noise_DataSet[:, 2], x=Noise_DataSet[:, 0], y=Noise_DataSet[:, 1])
plt.show()

w1_without_Noise, w2_without_Noise, bias_without_Noise = utils.PTA(Without_Noise_DataSet[:, :-1],
                                                                   Without_Noise_DataSet[:, -1])
x = np.linspace(-2, 2)
a = -w1_without_Noise / w2_without_Noise
y2 = a * x - bias_without_Noise / w2_without_Noise
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=Without_Noise_DataSet, hue=Without_Noise_DataSet[:, 2], x=Without_Noise_DataSet[:, 0],
                y=Without_Noise_DataSet[:, 1])
plt.show()

# Q4

w1_without_Noise_Fixed, w2_without_Noise_Fixed = utils.PTA_with_constant_bias(Without_Noise_DataSet[:, :-1],
                                                                              Without_Noise_DataSet[:, -1])
x = np.linspace(-2, 2)
a = -w1_without_Noise_Fixed / w2_without_Noise_Fixed
y2 = a * x
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=Without_Noise_DataSet, hue=Without_Noise_DataSet[:, 2], x=Without_Noise_DataSet[:, 0],
                y=Without_Noise_DataSet[:, 1])
plt.show()

# Q5

AND_Dataset = utils.AND_Data()
OR_Dataset = utils.OR_Data()
XOR_Dataset = utils.XOR_Data()

# AND

w1_without_Noise, w2_without_Noise, bias_without_Noise = utils.PTA(AND_Dataset[:, :-1], AND_Dataset[:, -1])
x = np.linspace(-2, 2)
a = -w1_without_Noise / w2_without_Noise
y2 = a * x - bias_without_Noise / w2_without_Noise
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=AND_Dataset, hue=AND_Dataset[:, 2], x=AND_Dataset[:, 0], y=AND_Dataset[:, 1])
plt.show()

w1_without_Noise, w2_without_Noise = utils.PTA_with_constant_bias(AND_Dataset[:, :-1], AND_Dataset[:, -1])
x = np.linspace(-2, 2)
a = 0
plt.axvline(x=a)
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=AND_Dataset, hue=AND_Dataset[:, 2], x=AND_Dataset[:, 0], y=AND_Dataset[:, 1])
plt.show()

# OR

w1_without_Noise, w2_without_Noise, bias_without_Noise = utils.PTA(OR_Dataset[:, :-1], OR_Dataset[:, -1])
x = np.linspace(-2, 2)
a = -w1_without_Noise / w2_without_Noise
y2 = a * x - bias_without_Noise / w2_without_Noise
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=OR_Dataset, hue=OR_Dataset[:, 2], x=OR_Dataset[:, 0], y=OR_Dataset[:, 1])
plt.show()

w1_without_Noise, w2_without_Noise = utils.PTA_with_constant_bias(OR_Dataset[:, :-1], OR_Dataset[:, -1])
x = np.linspace(-2, 2)
a = 0
plt.axvline(x=a)
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=OR_Dataset, hue=OR_Dataset[:, 2], x=OR_Dataset[:, 0], y=OR_Dataset[:, 1])
plt.show()

# XOR

w1_without_Noise, w2_without_Noise, bias_without_Noise = utils.PTA(XOR_Dataset[:, :-1], XOR_Dataset[:, -1])
x = np.linspace(-2, 2)
a = 0
if w2_without_Noise != 0:
    a = -w1_without_Noise / w2_without_Noise
y2 = a * x - bias_without_Noise / w1_without_Noise
plt.axvline(x=a)
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=XOR_Dataset, hue=XOR_Dataset[:, 2], x=XOR_Dataset[:, 0], y=XOR_Dataset[:, 1])
plt.show()

w1_without_Noise, w2_without_Noise = utils.PTA_with_constant_bias(XOR_Dataset[:, :-1], XOR_Dataset[:, -1])
x = np.linspace(-2, 2)
a = 0
if w2_without_Noise != 0:
    a = -w1_without_Noise / w2_without_Noise
y2 = a * x
plt.plot(x, y2, "k-")
sns.set(rc={'figure.figsize': (10, 8)})
sns.scatterplot(data=XOR_Dataset, hue=XOR_Dataset[:, 2], x=XOR_Dataset[:, 0], y=XOR_Dataset[:, 1])
plt.show()
