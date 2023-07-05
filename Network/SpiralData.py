#external libraries
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import numpy as np

coordinates, colors = spiral_data(samples = 1000, classes = 3)

plt.scatter(x = np.transpose(coordinates)[0],y = np.transpose(coordinates)[1], c = colors, s = 5)

plt.show()