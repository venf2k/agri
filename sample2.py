import h5py
import numpy as np
train_dataset = h5py.File('dataset_full_5000.hdf5', "r")
list(train_dataset.keys())
X_train_orig = np.array(train_dataset["img"][:]) # your 40.000 train set features: shape = (8 x 5.000,100,100,3)
train_set_y_orig = np.array(train_dataset["labels"][:])
Y_train_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T # your 1080 train set images reshaped (40.000 rows, each row 1 image): shape = (30.000, 40.000)


#import matplotlib.pyplot as plt
#index = 0
#%matplotlib inline
#plt.imshow(X_train_orig[index])
#print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
