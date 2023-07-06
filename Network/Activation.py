#external Libraries
import math
import numpy as np

def ReLU(x):
    return(np.maximum(0, x))

def dReLU(x):
    if x < 0:
        return(0)
    else:
        return(1)

def softmax(x):
    x = math.e**x
    return(x/ sum(x))

def dSoftmax(x):
    x = x.reshape(-1, 1)
    return(np.sum(np.diagflat(x) - np.matmul(x, x.T), axis = 0))