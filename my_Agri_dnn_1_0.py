##########################################################
#
# 3-LAYER DNN CLASSIFIER
#
##########################################################

import math
import numpy as np
import scipy.io

train_label=[]
train_X=[]
train_Y=[]

test_label=[]
test_X=[]

###########################################################
#
# 	UTILS
#
###########################################################

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def load_params_and_grads(seed=1):
    np.random.seed(seed)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
                    
    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
   #     assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
   #    assert(parameters['W' + str(l)].shape == layer_dims[l], 1)
        
    return parameters


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        ### END CODE HERE ###
        
    return parameters


def compute_cost(a3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    return cost

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    
    return a3, cache


def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    
    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    #print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

 
def load_train_query_dataset():
  ##init array
  train_label=[]
  train_X=[]
  train_Y=[]
  query_label=[]
  query_X=[]

  ##open file per test vs stdin
  ##open file per test vs stdin
  ##f=open('C:/Apps/Python/DS/ML/DL_Ad_train_query_data.txt')
  N, M = map(int, input().split())
  ##N, M = map(int, f.readline().split())
  #lettura file (training data) riga per riga
  for i in range(N):
      s=input().split()
      ##s=f.readline().split()
      train_label.append((s[0]))
      train_Y.append((s[1]))
      for j in range(M):
          idx=s[j+2].find(':')
          train_X.append((s[j+2][idx+1:]))

  Q= int(input())
  ##Q = int(f.readline())
  #lettura test data riga per riga
  for i in range(Q):
      s=input().split()
      ##s=f.readline().split()
      query_label.append((s[0]))
      for j in range(M):
          idx=s[j+1].find(':')
          query_X.append((s[j+1][idx+1:]))
  
  ##close file per test vs stdout
  ##f.close()
  
  train_label = np.array(train_label).reshape(1,N)
  query_label = np.array(query_label).reshape(1,Q)
  train_Y = np.array(train_Y, float).reshape(1,N)
  train_Y = np.where(train_Y == -1, 0, train_Y)
  train_X = np.array(train_X, float).reshape(N,M).transpose()
  query_X = np.array(query_X, float).reshape(Q,M).transpose()

  #standardizzazione dei dati
  for i in range(M):
    train_X[i,:]= (train_X[i,:] - np.mean(train_X[i,:]))/(np.std(train_X[i,:])+epsilon)  
    query_X[i,:]=(query_X[i,:] - np.mean(query_X[i,:]))/(np.std(query_X[i,:])+epsilon)

  return train_label, train_X, train_Y, query_label, query_X, N, Q
         

def print_predict(label,Y,Q):
  Y = np.where(Y == 0, "-1", "+1")
  for i in range(Q):
    print(label[0][i], Y[0][i])



###########################################################
# 3-LAYER DNN MODEL
###########################################################
def model(X, Y, layers_dims, optimizer="gd", learning_rate=0.0007, epsilon=1e-8, num_iterations=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost

    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    
    # Optimization loop
    for i in range(num_iterations):
        
        # Forward propagation
        a3, caches = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(a3, Y)

        # Backward propagation
        grads = backward_propagation(X, Y, caches)

        # Update parameters
        parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters


###########################################################
# MAIN
###########################################################
epsilon=1e-8
#load datasets
#train_X, train_Y = load_dataset()
train_label, train_X, train_Y, query_label, query_X, N, Q = load_train_query_dataset()
##train_label, train_X, train_Y, test_label, test_Y, test_X, N, T = load_train_test_dataset()

#train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
#parameters = model(train_X, train_Y, layers_dims, optimizer="gd", learning_rate=0.01)
parameters = model(train_X, train_Y, layers_dims, learning_rate=0.1, num_iterations=2500, print_cost=False)

# Predict
#predictions = predict(train_X, train_Y, parameters)
query_Y = predict(query_X, np.ones((1,Q), dtype=int), parameters)
##predictions = predict(test_X, test_Y, parameters)

#output prediction
print_predict(query_label, query_Y, Q)
##print_predict(test_label, test_Y, T)