import numpy as np


def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return result

def relu(z):
    result = max(0,z)
    return result

def initialize_weights(dimensions):
    parameters = {}
    for l in range(1,len(dimensions)):
        parameters["W"+str(l)] = np.random.randn(dimensions[l],dimensions[l-1])
        parameters["b"+str(l)] = np.zeros((dimensions[l],1))

    return parameters

def linear_forward(X,A_prev,W,b):
    Z = np.dot(W,X) + b
    linear_cache = {
        "A_prev":A_prev,
        "W":W,
        "b":b
    }
    return Z , linear_cache

def linear_activation_forward(X,Y,A_prev,activation):
    if activation == "relu":
        Z,cache = 
        A = 
        pass
    elif activation == "sigmoid":
        pass
    return A , activation_cache


dimensions = [3,4,1]
parameters = initialize_weights(dimensions)
print(parameters)