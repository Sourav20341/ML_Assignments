import gzip
import idx2numpy
import Encoder
from NeuralNetwork import NeuralNetwork
from Weight import *
import AccuracyScore
from ActivationFunction import *

# data importing
feature_train = gzip.open('train-images-idx3-ubyte.gz', 'r')
label_train = gzip.open('train-labels-idx1-ubyte.gz', 'r')
feature_test = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
label_test = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')

#  training and testing data
X_train = idx2numpy.convert_from_file(feature_train)
Y_train = idx2numpy.convert_from_file(label_train)
X_test = idx2numpy.convert_from_file(feature_test)
Y_test = idx2numpy.convert_from_file(label_test)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]*X_test.shape[2])
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
unique_y = np.unique(Y_train)
X_train = X_train/255
X_test = X_test/255

# Encode Y train and Y test
one_hot_encoder = Encoder.One_Hot_Encoder()
Y_train_encoded = one_hot_encoder.fit_transform(Y_train,unique_y)
Y_test_encoded = one_hot_encoder.fit_transform(Y_test,unique_y)

mlp_model = NeuralNetwork(4,(256,128,64,32),0.001,ReLU(),random_init,100,256)
Y_train = Y_train.reshape(len(Y_train),1)
mlp_model.fit(X_train,Y_train)
Y_pred = mlp_model.predict(X_test)
print(AccuracyScore.accuracy_score(Y_test,Y_pred))