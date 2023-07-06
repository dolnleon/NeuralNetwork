#external libraries
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import numpy as np

#internal libraries
from NeuralNetwork import Network

coordinates, colors = spiral_data(samples = 100, classes = 2)

plt.scatter(x = np.transpose(coordinates)[0],y = np.transpose(coordinates)[1], c = colors, s = 5)

net = Network([2, 5, 5, 2])

print("Output: ", net.forward(coordinates[1]))
print("Loss: ", net.cost(colors[1]))

plt.show()