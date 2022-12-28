# -*- coding: utf-8 -*-
"""Q3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mKNSFtBSG2wJcDPF3jxuhsCu0eEEzx2F

MOUNITNG DRIVE
"""

from google.colab import drive
drive.mount('/content/drive')

"""IMPORTING DATA"""

import pandas as pd
import numpy as np

trainingdata = pd.read_csv("/content/drive/MyDrive/fashion-mnist_train.csv")
testingdata = pd.read_csv("/content/drive/MyDrive/fashion-mnist_test.csv")

"""TRAINING DATA"""

from sklearn.model_selection import train_test_split
training_data_X = trainingdata.iloc[:,1:].values
training_data_Y = trainingdata.iloc[:,0].values
X_train,X_validate,Y_train,Y_validate = train_test_split(training_data_X,training_data_Y,test_size = 0.15,random_state = 1)
unique_labels = np.unique(Y_train)
X_train = X_train/255

"""TESTING DATA"""

X_test = testingdata.iloc[:,1:].values
Y_test = testingdata.iloc[:,0].values
X_test = X_test/255

"""Import Pacakages"""

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

Y_reshape = np.reshape(Y_train,(len(Y_train),1))
Y_reshape_val = np.reshape(Y_validate,(Y_validate.shape[0],1))

def shuffle(X,Y):
  data = np.column_stack([X, Y])
  np.random.shuffle(data)
  X_t = data[:,0:-1]
  Y_t = data[:,-1]
  return (X_t,Y_t)

"""Linear Model


"""

models_identity = MLPClassifier(hidden_layer_sizes = (256,32),activation = "identity",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_identity.partial_fit(X_t,Y_t,unique_labels)
  predict = models_identity.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict))
  
  validation_pred = models_identity.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_identity.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_identity.predict(X_test)))

"""Sigmoid Model"""

models_logistic = MLPClassifier(hidden_layer_sizes = (256,32),activation = "logistic",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_logistic.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_logistic.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_logistic.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score

print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_logistic.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_logistic.predict(X_test)))

"""Tanh Model"""

models_tanh = MLPClassifier(hidden_layer_sizes = (256,32),activation = "tanh",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_tanh.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_tanh.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_tanh.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_tanh.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_tanh.predict(X_test)))

"""ReLU Model"""

models_relu = MLPClassifier(hidden_layer_sizes = (256,32),activation = "relu",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""Best Accuracy we get on ReLU Model

Learning rate = 0.1
"""

models_relu = MLPClassifier(hidden_layer_sizes = (256,32),activation = "relu",batch_size = 128,learning_rate_init = 0.1)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""Learning rate = 0.01"""

models_relu = MLPClassifier(hidden_layer_sizes = (256,32),activation = "relu",batch_size = 128,learning_rate_init = 0.01)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""Learning rate = 0.001"""

models_relu = MLPClassifier(hidden_layer_sizes = (256,32),activation = "relu",batch_size = 128,learning_rate_init = 0.001)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""Q3

(128,16)
"""

models_relu = MLPClassifier(hidden_layer_sizes = (128,16),activation = "relu",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""(200,24)"""

models_relu = MLPClassifier(hidden_layer_sizes = (200,24),activation = "relu",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""(150,20)"""

models_relu = MLPClassifier(hidden_layer_sizes = (150,20),activation = "relu",batch_size = 128)
training_loss = []
validation_loss = []
for i in range(80):
  X_t,Y_t = shuffle(X_train,Y_reshape)
  models_relu.partial_fit(X_t,Y_t,unique_labels) 
  
  predict = models_relu.predict_proba(X_train)
  training_loss.append(log_loss(Y_reshape,predict,labels = unique_labels))
  
  validation_pred = models_relu.predict_proba(X_validate)
  validation_loss.append(log_loss(Y_reshape_val,validation_pred,labels = unique_labels))

plt.plot(training_loss,label = "Training loss",color = "green")
plt.plot(validation_loss,label = "Validation loss",color = "red")
plt.xlabel("Number of Epochs")
plt.title("Loss vs Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy Score For Validate = ",accuracy_score(Y_validate,models_relu.predict(X_validate)))
print("Accuracy Score For Test = ",accuracy_score(Y_test,models_relu.predict(X_test)))

"""GRID SEARCH"""

from sklearn.model_selection import GridSearchCV

model_MLP = MLPClassifier(max_iter = 200,early_stopping = True)
paramters = {
    "learning_rate_init" : [0.001,0.005],
    "hidden_layer_sizes" : [(256,32),(280,40)],
    "batch_size" : [256,128],
    "activation" : ["relu","logistic"]
}
gridSearch = GridSearchCV(model_MLP,paramters,n_jobs = -1,verbose = 3)
gridSearch.fit(X_train,Y_train)

print(gridSearch.best_params_)