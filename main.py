import sys 
import numpy as np # linear algebra
from scipy.stats import randint
from scipy.stats import stats
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import tensorflow as tf
import random
import os
import argparse
from utils import load_data, print_args

parser = argparse.ArgumentParser()
parser.add_argument("--num_compressed", default='100', type=int)
parser.add_argument("--num_trial", default='0', type=int)
parser.add_argument("--window_size", default='10', type=int)
parser.add_argument("--batch_size", default='20', type=int)
parser.add_argument("--epochs", default='1000', type=int)
parser.add_argument("--dataset", default='prec', type=str)


args = parser.parse_args()

'''
copy arguments from argparse
'''
num_compressed = args.num_compressed
num_trial = args.num_trial
window_size = args.window_size
batch_size = args.batch_size
epochs = args.epochs
dataset = args.dataset

state_size = num_compressed

'''
load data
'''
original_signal, sensors, n_rec, p_rec = load_data(dataset, num_compressed)

'''
create result folder
'''
path = "./results/"+dataset+"_"+str(num_compressed) + "_" + str(num_trial)
if not os.path.isdir(path+"/"):
  os.mkdir(path+"/")

'''
pre-processing data
'''
compressed_signal = original_signal[:,sensors]
if dataset =='prec':
  train_end = 700
  n_neurons_1 = 512
  n_neurons_2 = 1024
elif dataset =='sst':
  train_end = int(52*16)
  n_neurons_1 = 1024
  n_neurons_2 = 4096
data_train_input = compressed_signal[: train_end, :]
data_train_output = original_signal[: train_end, :]
data_test_input = compressed_signal[train_end:, :]
data_test_output = original_signal[train_end:, :]

train_X, train_y = data_train_input, data_train_output
test_X, test_y = data_test_input, data_test_output
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

input_state_size = np.size(compressed_signal,1)
output_size = p_rec

# reshape trainging/testing data according to sliding window
def window_stack(a, window_size=10):
    return np.array([a[i:i + window_size][::1] for i in range(0, len(a) - window_size+1)])
  
train_X_window = window_stack(train_X, window_size)
train_X_window = train_X_window[:,:,:]
train_X = train_X[window_size-1:,:]
print(train_X_window.shape)
train_y_window = train_y[window_size-1:,:]
print(train_y_window.shape)

test_X_window = window_stack(test_X, window_size)
test_X_window = test_X_window[:,:,:]
test_X = test_X[window_size-1:,:]
print(test_X_window.shape)
test_y_window = test_y[window_size-1:,:]
print(test_y_window.shape)


'''
constructing model
'''
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, window_size, input_state_size])
X_current = tf.placeholder(dtype=tf.float32, shape=[None, input_state_size])
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

W_hidden_1 = tf.Variable(weight_initializer([input_state_size+state_size, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_2, p_rec]))
bias_out = tf.Variable(bias_initializer([p_rec]))

# Forward pass
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# state.h.shape: (None, state_size)
lstm_cell_out = tf.concat(axis=1,values=[state.h, X_current])
hidden_1 = tf.nn.relu(tf.add(tf.matmul(lstm_cell_out, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

# model output
out = tf.add(tf.matmul(hidden_2, W_out), bias_out)
# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y)) #mse, element wise squred difference
_,variance = tf.nn.moments(tf.reduce_mean(tf.squared_difference(out, Y), axis = 1), axes = [0])


''' Optimizer'''
opt = tf.train.GradientDescentOptimizer(0.5).minimize(mse)
init_op = tf.global_variables_initializer()

# Run
mse_train = []
mse_test = []
variance_test = []

original_index = [o for o in range(len(train_y_window))]
shuffle_indices_for_training_data = np.array(original_index)

with tf.Session() as sess:
  sess.run(init_op)
  for e in range(epochs):
    print ('Epoch ', e, ' /', epochs)
    for i in range(0, len(train_y_window) // batch_size):
      start = i * batch_size
      batch_x = train_X_window[start:start + batch_size,:,:]
      batch_y = train_y_window[start:start + batch_size,:]
      batch_x_current = train_X[start:start + batch_size,:]
      # Run optimizer with batch
      sess.run(opt, feed_dict={X: batch_x, Y: batch_y, X_current: batch_x_current})
      # Show progress
      # if np.mod(i, 50) == 0:
      if np.mod(i, 50) == 0:
        # MSE train and test
        mse_train.append(sess.run(mse, feed_dict={X: train_X_window, Y: train_y_window, X_current: train_X}))
        mse_test.append(sess.run(mse, feed_dict={X: test_X_window, Y: test_y_window, X_current: test_X}))
        variance_test.append(sess.run(variance, feed_dict={X: test_X_window, Y: test_y_window, X_current: test_X}))
        print('MSE Train: ', mse_train[-1], '; MSE Test: ', mse_test[-1],'; var: ', variance_test[-1])
      
  pred = sess.run(out, feed_dict={X: test_X_window, Y: test_y_window, X_current: test_X})
  # np.savetxt(path+"/random_bench_sensors.csv", sensors, delimiter=",")
  np.savetxt(path+"/mse_test.csv", mse_test, delimiter=",")
  np.savetxt(path+"/mse_train.csv", mse_train, delimiter=",")
  np.savetxt(path+"/mse_var.csv", variance_test, delimiter=",")
  if num_compressed ==100 and num_trial == 1:
        np.savetxt(path+"/pred.csv", pred, delimiter=",")