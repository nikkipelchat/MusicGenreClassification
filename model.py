# -*- coding: utf-8 -*-

import numpy as np

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def createModel(nbClasses, imageSizeX, imageSizeY):
  print("[+] Creating model...")
  convnet = input_data(shape=[None, imageSizeX, imageSizeY, 1], name='input')

  convnet = conv_2d(convnet, 64, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = conv_2d(convnet, 128, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = conv_2d(convnet, 256, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = conv_2d(convnet, 512, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = conv_2d(convnet, 1024, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = conv_2d(convnet, 2048, 2, activation='relu', weights_init="Xavier")
  convnet = max_pool_2d(convnet, 2)

  convnet = fully_connected(convnet, 4096, activation='relu')
  convnet = dropout(convnet, 0.5)

  convnet = fully_connected(convnet, nbClasses, activation='softmax')
  convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy')

  model = tflearn.DNN(convnet, tensorboard_dir='tensorboard', tensorboard_verbose=3)
  print("    Model created!")
  return model