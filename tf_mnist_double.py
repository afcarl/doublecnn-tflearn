#!/usr/bin/env python

"""MNIST with TFlearn."""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from double_cnn import conv_2d_double

# Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
from mnist_data import load_data
X, Y, testX, testY = load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d_double(network, 32, 3, activation='relu')
# network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 50, 3, activation='relu')
# network = tflearn.layers.conv.global_avg_pool(network, name='gap')
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.5)
# network = tflearn.layers.core.reshape(network, [-1, 10], name='Reshape')
# network = tflearn.layers.core.activation(network, activation='softmax')
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
          validation_set=({'input': testX}, {'target': testY}),
          snapshot_step=100, show_metric=True, run_id='convnet_mnist')
