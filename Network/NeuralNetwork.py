#external libraries
import math
import numpy as np

#internal libraries
from Activation import ReLU
from Layers import Layer
from Layers import LayerDense
from Layers import LayerOutput

class Network():
    def __init__(self, neurons):
        #an array with the sizes of layers
        self.neurons = neurons
        self.layers = []
        for i in range(len(neurons) - 2):
            self.layers.append(LayerDense(neurons[i], neurons[i + 1]))
        self.layers.append(LayerOutput(neurons[len(neurons) - 2], neurons[len(neurons) - 1]))
       

    def forward(self, inputs):
        self.layers[0].forward(inputs)
        for i in range (len(self.neurons) - 2):
            self.layers[i + 1].forward(self.layers[i].outputs)

    def backward(self):
        pass      

    #Functions for debuging.
    #Will be removed in the final version.    

    #This function prints the values of the neurons in the network, withous the input neurons.
    def Values(self):
        for i in self.layers:
            print(i.outputs)  