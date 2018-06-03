################################################################################
#
# M A I N
#
# DIR = /media/user/_home1/apps/python/DL/Agri
#
################################################################################

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tf_utils import load_dataset
from sklearn.svm import LinearSVC
%matplotlib inline

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train = X_train_orig.reshape(X_train_orig.shape[0], -1)
X_test = X_test_orig.reshape(X_test_orig.shape[0], -1)
Y_train = Y_train_orig[0]
Y_test = Y_test_orig[0]

# Normalize image vectors
X_train = X_train / 255.
X_test = X_test / 255.


svm=LinearSVC()
svm.fit(X_train, Y_train)
