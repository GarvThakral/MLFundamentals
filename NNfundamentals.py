# Neural Network with 2 layers
import numpy as np

def sigmoid(Z):
    result = 1/(1+np.exp(-Z))
    return result

def initialize_parameters(n_x,n_h,n_y):
    # Initializing Parameters
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters 

def forward_propagation(X,Y,parameters,train = True):
    # Fetching Parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Calculating the linear function and activation for layer 1
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)

    # # Calculating the linear function and activation for layer 1
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    if train:
        cost = compute_cost(A2,Y)
    else:
        cost = 0
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2,
    }
    return cache , cost 

def compute_cost(Y_hat,Y):
    # X shape
    m = Y.shape[1]
    cost = -1/m*np.sum(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))
    return cost

def backward_propagation(X,Y,cache,parameters):
    # X shape 
    m = X.shape[1]

    # Initializing cache values
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Fetching Parameters
    W2 = parameters["W2"]

    # Calculating dW and dB
    dZ2 = A2 - Y
    dW2 = 1/m*np.dot(dZ2,A1.T)
    dB2 = 1/m*np.sum(dZ2,axis = 1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1/m*np.dot(dZ1,X.T)
    dB1 = 1/m*np.sum(dZ1,axis = 1,keepdims=True)

    grads = {
        "dW1":dW1,
        "dW2":dW2,
        "dB1":dB1,
        "dB2":dB2,
    }
    return grads

def update_parameters(parameters,grads,learning_rate):
    # Fetching parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Fetching Gradients 
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    dB1 = grads["dB1"]
    dB2 = grads["dB2"]

    W1 -= learning_rate*dW1 
    W2 -= learning_rate*dW2 
    b1 -= learning_rate*dB1 
    b2 -= learning_rate*dB2 

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

def train(X,Y,iterations,learning_rate):
    params = initialize_parameters(3,4,1)
    costArr = []
    for i in range(0,iterations):
        cache,cost = forward_propagation(X,Y,params)
        costArr.append(cost)
        grads = backward_propagation(X,Y,cache,params)
        params = update_parameters(params,grads,learning_rate)
        if i%100 == 0:
            print(f"Iteration {i} — Cost: {cost}")
    tuned_weights = {
        "W1":params["W1"],
        "W2":params["W2"],
        "b1":params["b1"],
        "b2":params["b2"]
    }
    return tuned_weights
def predict(X,params):
    cache , cost = forward_propagation(X = X,Y = 0,parameters=params,train = False)
    return cache["A2"]
def model():
    pass

# Initializing Values
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
Y = np.array([[1,0,1]])
tuned_weights = train(X,Y,2000,0.01)
result = predict(np.array([[2],[5],[8]]),params=tuned_weights)
result = result[0] >= 0.5
print(result)