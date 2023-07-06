#external Libraries
import math
import numpy as np

#internal libraries
from NeuralNetwork import Network

net = Network([2, 6, 5, 3])

inputs = np.array([6, 3])

net.forward(inputs)

net.Values()