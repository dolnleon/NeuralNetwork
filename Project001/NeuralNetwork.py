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
        print(self.layers)

    def forward(self, inputs):
        pass

    def backward(self):
        pass            

net = Network([2, 6, 5, 3])

inputs = np.array([6, 3])