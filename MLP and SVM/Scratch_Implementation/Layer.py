import numpy as np

class Layer:

    def __init__(self,activation_func,weight_init_func,number_of_input,number_of_output):
        self.activation_func = activation_func
        self.weights = weight_init_func.func(number_of_input,number_of_output)
        self.number_of_input = number_of_input
        self.number_of_output = number_of_output
        self.act_out = []
        self.wTx = []
        self.bias = np.random.rand(1, number_of_output)*0.1

    def forward_propagation(self, x):
        self.x = x
        self.wTx = np.dot(self.x, self.weights) + self.bias
        self.act_out = self.activation_func.func(self.wTx)
        return self.act_out

    def backward_propagation(self,error,learning_rate,output_layer = False):
        delta = self.activation_func.derivative(self.wTx) * error
        input_error = np.dot(delta, self.weights.T)
        w_error = np.dot(self.x.T,delta)
        self.weights -= learning_rate * w_error
        self.bias -= learning_rate * delta
        return input_error


