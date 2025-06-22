import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def relu(Z):
    result = np.maximum(0,Z)
    return result , Z

def sigmoid(Z):
    result = 1/(1+np.exp(-Z))
    return result , Z

def relu_backward(dA,activation_cache):
    Z = activation_cache 
    dZ = dA * (Z > 0)
    return dZ

def sigmoid_backward(dA,activation_cache):
    Z = activation_cache 
    A = 1 / (1 + np.exp(-Z)) 
    dZ = dA * A * (1 - A)
    return dZ

def initialize_parameters_deep(layers_dims):
    parameters = {}
    for l in range(1,len(layers_dims)):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*0.01
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
    return parameters

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b
    linear_cache = (A_prev,W,b)
    return Z , linear_cache

def linear_activation_forward(A_prev, W , b , activation):
    linear_cache = None
    activation_cache = None
    A = None
    if activation == "relu":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)

    cache = (linear_cache , activation_cache)

    return A , cache

def L_model_forward(X,Y,parameters,lambd):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A , cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    AL , cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    cost = compute_cost(AL,Y,lambd,parameters)
    return AL , cost , caches 

def compute_cost(Y_hat,Y,lambd,parameters):
    m = Y.shape[1]
    epsilon = 1e-8
    L = len(parameters)//2
    result = -1/m*(np.sum(Y*np.log(Y_hat + epsilon)+(1-Y)*np.log(1 - Y_hat + epsilon)))
    regularization_term = 0
    for l in range(1, L+1):
        regularization_term += np.sum(np.square(parameters["W"+str(l)]))
    regularization_term = (lambd / (2 * m)) * regularization_term
    total_cost = result + regularization_term
    return total_cost

def linear_backward(dZ, linear_cache,lambd):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,A_prev.T) + (lambd/m)*W
    db = 1/m*np.sum(dZ,axis = 1,keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev,dW,db

def backward_linear_activation(dA , current_cache , activation,lambd):
    linear_cache,activation_cache = current_cache
    dA_prev = None
    dW = None
    db = None
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache,lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache,lambd)
    return dA_prev , dW , db

def L_model_backward(AL,Y,caches,lambd):
    L = len(caches)
    grads = {}
    Y = Y.reshape(AL.shape)
    current_caches = caches[L-1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev , dW , db = backward_linear_activation(dAL,current_caches,"sigmoid",lambd)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    for i in reversed(range(L-1)):
        current_caches = caches[i]
        dA_prev , dW , db = backward_linear_activation(dA_prev,current_caches,"relu",lambd)    
        grads["dW"+str(i+1)] = dW
        grads["db"+str(i+1)] = db
    return grads

def update_parameters(parameters , grads , learning_rate):
    # Computing length of parameters
    L = len(parameters)//2
    
    for l in range(1,L+1):
        parameters["W"+str(l)] -= learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate*grads["db"+str(l)]
    return parameters

def L_model_full(X,Y,num_iterations,learning_rate,lambd):
    layers_dims = [3, 16, 8, 1]
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL,cost,caches = L_model_forward(X,Y,parameters,lambd)
        grads = L_model_backward(AL,Y,caches,lambd)
        parameters = update_parameters(parameters,grads,learning_rate)
        if i%100 == 0:
            print(cost)
    return parameters


def L_model_forward_no_cost(X, parameters):
    A = X
    caches = []
    L = len(parameters)//2
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")
    caches.append(cache)
    return AL, caches


def predict(X, parameters):
    """
    Arguments:
    X          -- input data of size (n_x, m)
    parameters -- python dictionary containing your trained parameters
    
    Returns:
    preds      -- vector of predictions of size (1, m), values are 0 or 1
    """
    # Forward propagation
    AL , _ = L_model_forward_no_cost(X, parameters) # we ignore Y and caches here
    
    # Convert probabilities to 0/1 predictions
    preds = (AL > 0.5).astype(int)
    return preds




data = load_breast_cancer()
X_full, Y_full = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X_full[:, [0,1,2]]).T
Y = Y_full.reshape(1, -1)



# Train
parameters = L_model_full(X, Y, num_iterations=20000, learning_rate=0.1,lambd=0.5)

# Predict
predictions = predict(X, parameters)

accuracy = np.mean(predictions == Y) * 100
print(f"Train accuracy: {accuracy:.2f}%")
