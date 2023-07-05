#external Libraries
import math
import numpy as np

def ReLU(x):
    return(np.maximum(0, x))

def softmax(x):
    x = math.e**x
    return(x/ sum(x))