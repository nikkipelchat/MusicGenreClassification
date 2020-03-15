'''Create a DNN model'''
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import keras
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import rmsprop
# import numpy as np

def createModelUsingTensorflow(nbClasses, imageSizeX, imageSizeY):
  '''Create the Deep Neural Network Model'''
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

  # model = tflearn.DNN(convnet, tensorboard_dir='tensorboard', tensorboard_verbose=3)
  model = tflearn.DNN(convnet)
  print("    Model created!")
  return model


def createModelUsingKeras(nbClasses, imageSizeX, imageSizeY):
  '''Create the Deep Neural Network Model'''
  print("[+] Creating model...")
  if K.image_data_format() == 'channels_first':
      input_shape = (1, imageSizeX, imageSizeY) # might have X and Y mixed up
  else:
      input_shape = (imageSizeX, imageSizeY, 1) # might have X and Y mixed up

  model = Sequential()
  
  model.add(ZeroPadding2D((1,1), input_shape=input_shape))

  model.add(Conv2D(128, 2, activation='sigmoid', kernel_initializer="glorot_normal"))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(256, 2, activation='sigmoid', kernel_initializer="glorot_normal"))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(512, 2, activation='sigmoid', kernel_initializer="glorot_normal"))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(1024, 2, activation='sigmoid', kernel_initializer="glorot_normal"))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(2048, 2, activation='sigmoid', kernel_initializer="glorot_normal"))
  model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())
  model.add(Dense(4096))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(nbClasses))
  model.add(Activation('softmax'))
  opt = rmsprop()
  model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

  print("    Model created!")
  return model
