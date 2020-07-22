from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import scipy.io
import h5py

from ladder_net import get_ladder_network_fc


# get the dataset
inp_size = 64*64 # size of mnist dataset 
n_classes = 4
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

CellData_train = h5py.File("CellData_train.mat")
x_train = np.transpose(np.asarray(CellData_train['x_train'],dtype='float32'),(1,0))
y_train = np.transpose(np.asarray(CellData_train['y_train'],dtype='int32'),(1,0))
y_train = np.squeeze(y_train)
y_train = y_train-1
print(x_train.shape)
print(y_train.shape)

CellData_test = h5py.File("CellData_test.mat")
x_test = np.transpose(np.asarray(CellData_test['x_test'],dtype='float32'),(1,0))
y_test = np.transpose(np.asarray(CellData_test['y_test'],dtype='int32'),(1,0))
y_test = np.squeeze(y_test)
y_test = y_test-1

y_train = keras.utils.to_categorical(y_train, n_classes)
#y_test  = keras.utils.to_categorical(y_test,  n_classes)

# only select 100 training samples 
idxs_annot = range(x_train.shape[0])
random.seed(0)
idxs_annot = np.random.choice(x_train.shape[0], 4000)

x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)

# initialize the model 
model = get_ladder_network_fc(layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes])

# train the model for 100 epochs
for _ in range(20):
    model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1)

y_test_pr = model.test_model.predict(x_test, batch_size=100)
scipy.io.savemat('test.mat', {'feature':y_test_pr.argmax(-1)})
#print("Test accuracy : %f" % accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))


