################################################################################
#
# M A I N
#
# ################################################################################

#cd  /media/user/_home1/apps/python/DL/Agri
#cd  D:/Apps/Python/DL/Agri
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

# FIT Model
svm=LinearSVC()
svm.fit(X_train, Y_train)

#PREDICT test set
test_label=svm.predict(X_test)


j = 0 
for i in range(Y_test.size):
  if test_label[i] == Y_test[i]: 
    j += 1

acc = j / i
print("Accurance: ", acc)