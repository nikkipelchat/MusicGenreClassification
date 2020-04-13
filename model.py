'''Create a DNN model'''
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# import keras
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import rmsprop

from config import checkpointPath
from imageFilesTools import createFolder

def createModelUsingTensorflow(nbClasses, imageSizeX, imageSizeY, imageSizeZ, args):
  '''Create the Deep Neural Network Model'''
  print("[+] Creating model...")
  convnet = input_data(shape=[None, imageSizeX, imageSizeY, imageSizeZ], name='input')

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
  createFolder(checkpointPath)
  model = tflearn.DNN(convnet, checkpoint_path='{}/model.tfl'.format(checkpointPath), max_checkpoints=1)

  if args.resume and args.epochs:
    try:
      model.load('{}/model.tfl-{}'.format(checkpointPath, args.resume))
      print("    Model retrieved and resuming training!")
    except Exception as err:
      print("Couldn't load the previous model", err)
      raise err
  else:
    print("    Model created!")
  return model


def createModelUsingKeras(nbClasses, imageSizeX, imageSizeY, imageSizeZ):
  '''Create the Deep Neural Network Model'''
  print("[+] Creating model...")
  if K.image_data_format() == 'channels_first':
    inputShape = (1, imageSizeX, imageSizeY) # might have X and Y mixed up
  else:
    inputShape = (imageSizeX, imageSizeY, imageSizeZ) # might have X and Y mixed up

  model = Sequential()

  model.add(ZeroPadding2D((1, 1), input_shape=inputShape))

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
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  print("    Model created!")
  return model
