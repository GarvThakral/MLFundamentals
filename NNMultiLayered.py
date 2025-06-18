import numpy as np


def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return result, z

def relu(z):
    result = np.maximum(0,z)
    return result,z

def initialize_weights(dimensions):
    parameters = {}
    for l in range(1,len(dimensions)):
        parameters["W"+str(l)] = np.random.randn(dimensions[l],dimensions[l-1])
        parameters["b"+str(l)] = np.zeros((dimensions[l],1))

    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    
    return Z , cache

def linear_activation_forward(A_prev,W,b,activation):
    linear_cache = None
    activation_cache = None
    A = None
    if activation == "relu":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = relu(Z) 
        
    elif activation == "sigmoid":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)
    
    cache = (linear_cache,activation_cache)
    return A , cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation = "relu")
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation = "sigmoid")
    caches.append(cache)
    return AL,caches

        

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost  = -1/m*np.sum((Y*np.log(AL)+(1-Y)*np.log(1-AL)))
    return cost

def linear_activation_backward(dA , cache ,activation):
    linear_cache , activation_cache = cache 
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)    
        dA_prev,dW,db = linear_backward(dZ,cache)
    elif activation == "relu":
        dZ = relu_backward(dA,activation_cache)   
        dA_prev,dW,db = linear_backward(dZ,cache) 
    return dA_prev , dW , db

def linear_backward(dZ, cache):
    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis = 1,keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev,dW,db


def relu_backward(dA,activation_cache):
    Z = activation_cache
    dZ = dA * (Z > 0)
    return dZ

def sigmoid_backward(dA,activation_cache):
    Z = activation_cache 
    A = 1 / (1 + np.exp(-Z)) 
    dZ = dA * A * (1 - A)
    return dZ

def update_parameters(parameters,grads,learnin_rate):
    L = len(parameters) // 2
    for l in range(1,L):
        parameters["W"+str(l)] -= learnin_rate*grads["W"+str(l)]
        parameters["b"+str(l)] -= learnin_rate*grads["b"+str(l)]
    return parameters
def L_model_backward(AL,Y,caches):
    A = AL
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    current_cache = caches[L-1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu") 
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

dimensions = [3,4,1]

parameters = initialize_weights(dimensions)
print(parameters)

linear_forward(A,W,b)