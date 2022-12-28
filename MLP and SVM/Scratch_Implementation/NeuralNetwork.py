import math
from Layer import *
import ActivationFunction


class NeuralNetwork:

    def __init__(self, number_of_layers, hidden_layers_sizes, learning_rate, Activation_func, Weight_init_func, epochs,
                 batch_size):
        self.number_of_layers = number_of_layers
        self.layer_sizes = hidden_layers_sizes
        self.lr = learning_rate
        self.activation = Activation_func
        self.weight_init = Weight_init_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.Layers = []
        self.initializeLayers()

    def initializeLayers(self):
        self.Layers.append(Layer(ActivationFunction.Linear(), self.weight_init, 784, 256))
        for i in range(self.number_of_layers - 1):
            self.Layers.append(Layer(self.activation, self.weight_init, self.layer_sizes[i], self.layer_sizes[i + 1]))

        self.Layers.append(Layer(self.activation, self.weight_init, self.layer_sizes[-1], 10))

    def predict_proba(self, X):
        output = X
        for i in range(len(self.Layers)):
            output = self.Layers[i].forward_propogation(output)

        return output

    def fit(self, X_train, Y_train):
        for i in range(self.epochs):
            loss = 0
            for j in range(0, X_train.shape[0], self.batch_size):
                X_t = X_train[j:j + self.batch_size]
                Y_t = Y_train[j:j + self.batch_size]
                for k in range(len(Y_t)):
                    x = X_t[k]
                    for layer in self.Layers:
                        x = layer.forward_propagation(x)

                    error = self.cross_entropy_loss(Y_t[k], x)
                    loss += error
                    for layer in range(len(self.Layers) - 1, 0, -1):
                        error = self.Layers[layer].backward_propagation(error, self.lr)

            loss /= len(Y_train)
            print(loss)

    def cross_entropy_loss(self, Y_true, Y_predict):
        loss = 0
        for i in range(len(Y_true)):
            loss -= math.log(Y_predict[i][Y_true[i] - 1] + 1e-15)
        return loss

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)

    def score(self, X, Y):
        y_pred = self.predict(X)
        score = 0
        for i in range(len(Y)):
            if y_pred[i] == Y[i]:
                score += 1
        return score / len(Y)
