#
# CONST HARD CODED
#
# NUM TRAIN EXAMPLES: 28.000
# NUM MINI BATCHES: 70
# PICTURE DIM: 30.000
#

import h5py
import numpy as np
import tensorflow as tf
import math


def load_dataset():
#
# load, split & shuffle 40.000 tuples dataset
#
    Dataset = h5py.File('data/dataset_full_5000.hdf5', "r")
    #list(Dataset.keys())
    X_Dataset_orig = np.array(Dataset["img"][:]) # your 40.000 train set features: shape = (8 x 5.000,100,100,3)
    Y_Dataset_orig = np.array(Dataset["labels"][:], int)
    Y_Dataset_orig = Y_Dataset_orig.reshape((1, Y_Dataset_orig.shape[0]))
    #Shuffle (X, Y)
    seed=0
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X_Dataset_orig.shape[0]                  # number of training examples
    permutation = list(np.random.permutation(m))
    X_Dataset_shuffled = X_Dataset_orig[permutation, :]
    Y_Dataset_shuffled = Y_Dataset_orig[:, permutation]
    X_Dataset_train = X_Dataset_shuffled[:28000,:]
    Y_Dataset_train = Y_Dataset_shuffled[:,:28000]
    X_Dataset_test = X_Dataset_shuffled[28000:,:]
    Y_Dataset_test = Y_Dataset_shuffled[:,28000:]
    classes = np.array(range(7), int)
    return X_Dataset_train, Y_Dataset_train, X_Dataset_test, Y_Dataset_test, classes


def random_mini_batches(X, Y, mini_batch_size = 70, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
        
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    #permutation = list(np.random.permutation(m))
    #shuffled_X = X[:, permutation]
    #shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    shuffled_X = X
    shuffled_Y = Y.reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [30000, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
    

def load_sample_dataset():
#
# load, split & shuffle 40.000 tuples dataset
#
    Dataset = h5py.File('data/dataset_full_5000.hdf5', "r")
    #list(Dataset.keys())
    X_Dataset_orig = np.array(Dataset["img"][:]) # your 40.000 train set features: shape = (8 x 5.000,100,100,3)
    Y_Dataset_orig = np.array(Dataset["labels"][:], int)
    Y_Dataset_orig = Y_Dataset_orig.reshape((1, Y_Dataset_orig.shape[0]))
    #Shuffle (X, Y)
    seed=0
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X_Dataset_orig.shape[0]                  # number of training examples
    permutation = list(np.random.permutation(m))
    X_Dataset_shuffled = X_Dataset_orig[permutation, :]
    Y_Dataset_shuffled = Y_Dataset_orig[:, permutation]
    X_Dataset_train = X_Dataset_shuffled[:2800,:]
    Y_Dataset_train = Y_Dataset_shuffled[:,:2800]
    X_Dataset_test = X_Dataset_shuffled[2800:3000,:]
    Y_Dataset_test = Y_Dataset_shuffled[:,2800:3000]
    classes = np.array(range(7), int)
    return X_Dataset_train, Y_Dataset_train, X_Dataset_test, Y_Dataset_test, classes


def load_sample_class_dataset(nclass = 0):
#
# load, split & shuffle 40.000 tuples dataset
#
    Dataset = h5py.File('data/dataset_full_5000.hdf5', "r")
    #list(Dataset.keys())
    X_Dataset_orig = np.array(Dataset["img"][:]) # your 40.000 train set features: shape = (8 x 5.000,100,100,3)
    Y_Dataset_orig = np.array(Dataset["labels"][:], int)
    Y_Dataset_orig = Y_Dataset_orig.reshape((1, Y_Dataset_orig.shape[0]))
    #Extract class (X, Y)
    X_Dataset_train = X_Dataset_orig[nclass*5000:nclass*5000+750,:]
    Y_Dataset_train = Y_Dataset_orig[:,nclass*5000:nclass*5000+750]
    X_Dataset_test = X_Dataset_orig[nclass*5000+750:nclass*5000+1000,:]
    Y_Dataset_test = Y_Dataset_orig[:,nclass*5000+750:nclass*5000+1000]
    classes = np.array(range(7), int)
    return X_Dataset_train, Y_Dataset_train, X_Dataset_test, Y_Dataset_test, classes


def show_images(images, cols = 3, titles = None):
    """Display a list of images in a single figure with matplotlib.
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()