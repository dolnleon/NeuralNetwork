#external libraries
import math
import numpy as np

#internal libraries
from Activation import ReLU
from Activation import softmax

#This class represent on elayer in a network, it takes in the number of inputs (n. of inputs in the previous layer) and the number of neurons in this layer.
class Layer:
    def __init__(self, n_inputs, n_neurons):
        #A matrix wich stores the weights of a single layer -> dim 2
        self.weights = np.random.rand(n_neurons, n_inputs)
        #A matrix that stores the biases of a single layer -> dim 1
        self.biases = np.random.rand(n_neurons)
        #A matrix with the values of neurons in this layer -> dim 1
        self.outputs = np.zeros(n_neurons)

#This subclass represents one dense layer in a nerual network.
class LayerDense(Layer):
    #This function initializes the forward pass of a single layer
    def forward(self, inputs):
        #A matrix with the values of neurons in this layer -> dim 1
        self.outputs = ReLU(np.matmul(self.weights, inputs) + self.biases)
    
    def backward(self):
        pass

#This subclass represents the last layer in a nerual network.
class LayerOutput(Layer):
    def forward(self, inputs):
        #A matrix with the values of neurons in this layer -> dim 1
        self.outputs = softmax(np.matmul(self.weights, inputs) + self.biases)

    def backward(self):
        pass