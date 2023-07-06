#external libraries
import numpy as np

#internal libraries
from Activation import ReLU
from Activation import softmax

a = softmax(np.array([1, 5, 2, 5]))

a = a.reshape(-1, 1)

print(np.sum(np.matmul(a, a.T), axis = 0))