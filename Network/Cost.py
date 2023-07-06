#external libraries
import math
import numpy as np

def cost(output, goal):
    output[goal] -= 1
    return(sum(output**2))

def dCost(output, goal):
    output[goal] -= 1
    return(2 * output)