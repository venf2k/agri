import h5py
import numpy as np
#
# Load Dataset
#
Dataset = h5py.File('data/dataset_full_5000.hdf5', "r")
list(Dataset.keys())
X_Dataset_orig = np.array(Dataset["img"][:]) # your 40.000 train set features: shape = (8 x 5.000,100,100,3)
Y_Dataset_orig = np.array(Dataset["labels"][:])
Y_Dataset_orig = Y_Dataset_orig.reshape((1, Y_Dataset_orig.shape[0]))


#Shuffle (X, Y)
seed=0
np.random.seed(seed)            # To make your "random" minibatches the same as ours
m = X_Dataset_orig.shape[0]                  # number of training examples
        
permutation = list(np.random.permutation(m))
X_Dataset_shuffled = X_Dataset_orig[permutation, :]
Y_Dataset_shuffled = Y_Dataset_orig[:, permutation]


X_Dataset_train = X_Dataset_shuffled[:30000,:]
Y_Dataset_train = Y_Dataset_shuffled[:,:30000]
X_Dataset_test = X_Dataset_shuffled[30000:,:]
Y_Dataset_test = Y_Dataset_shuffled[:,30000:]


# Flatten the training and test images
#X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T # your 1080 train set images reshaped (40.000 rows, each row 1 image): shape = (30.000, 40.000)



import matplotlib.pyplot as plt
index = 0
%matplotlib inline
plt.imshow(X_Dataset_shuffled[index])
print ("y = " + str(np.squeeze(Y_Dataset_shuffled[:, index])))
