# -*- coding: utf-8 -*-
import random
import string
import os
import time
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only log errors

from model import createModel
from datasetTools import getDataset
from config import slicesPath
from config import batchSize
from config import filesPerGenreMap
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceSize, sliceXSize, sliceYSize
from config import ignoreGenres

from songToData import createSlicesFromAudio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","slice"])
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Batch size: {}".format(batchSize))

if "slice" in args.mode:
  createSlicesFromAudio()
  sys.exit()

#List genres
allGenres = os.listdir(slicesPath)
allGenres = [filename for filename in allGenres if os.path.isdir(slicesPath+filename)]
setOfAllGenres = set(allGenres)
setOfGenresToIgnore = set(ignoreGenres)
#Genres to use
genres = setOfAllGenres - setOfGenresToIgnore
print("| Genres: {}".format(genres))
nbClasses = len(genres)

print("| Number of classes: {}".format(nbClasses))
print("| Slices per genre map: {}".format(filesPerGenreMap))
print("| Slice size: {}x{}".format(sliceXSize, sliceYSize))
print("--------------------------")

#Create model
model = createModel(nbClasses, sliceXSize, sliceYSize)

if "train" in args.mode:
  #Create or load new dataset
  train_X, train_y, validation_X, validation_y = getDataset(filesPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio, mode="train")

  #Define run id for graphs
  run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

  #Train the model
  print("[+] Training the model...")
  t0 = time.gmtime()
  history = model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
  t1 = time.gmtime()
  secondsToTrain = time.mktime(t1) - time.mktime(t0)
  hours, minutes = divmod(secondsToTrain, 3600)
  print("[+]: Time to train: {} hours, {:.0f} minutes".format(hours, minutes/60))
  print("[+] History: {}".format(history))
  print("    Model trained!")

  #Save trained model
  print("[+] Saving the weights...")
  model.save('musicDNN.tflearn')
  print("[+] Weights saved!")

  print("[+] Test Neural Network")
  test_X, test_y = getDataset(filesPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio, mode="test")

  print("[+] Loading weights...")
  model.load('musicDNN.tflearn')
  print("    Weights loaded!")
  #Evaluate 2
  testAccuracy = model.evaluate(test_X, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(testAccuracy))



if "test" in args.mode:
  #Create or load new dataset
  test_X, test_y = getDataset(filesPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio, mode="test")

  #Predict and compare
  #Prediction = model.predict_label(test_X)
  # actual = test_y
  # print("Length of TestX: {}".format(len(test_X)))
  # print("Prediction: {}".format(Prediction))
  # print("Actual: {}".format(actual))

  #Load weights
  print("[+] Loading weights...")
  model.load('musicDNN.tflearn')
  print("    Weights loaded!")

  # Run the model on one example
  # prediction = model.predict([test_X[0]])
  # print("Prediction: %s" % str(prediction[0][:8]))

  #Evaluate 1
  # predictions = model.predict(test_X)#[test_X[0], test_X[1], test_X[2], test_X[3], test_X[4], test_X[5], test_X[6], test_X[7], test_X[8], test_X[9], test_X[10]])
  # accuracy = 0
  # for prediction, actual in zip(predictions, test_y): #[test_y[0], test_y[1], test_y[2], test_y[3], test_y[4], test_y[5], test_y[6], test_y[7], test_y[8], test_y[9], test_y[10]]):
  # 	predicted_class = np.argmax(prediction)
  # 	actual_class = np.argmax(actual)
  # 	print("Predicted: {} and actual: {}".format(predicted_class, actual_class))
  # 	if(predicted_class == actual_class):
  # 		accuracy+=1

  # accuracy = accuracy / len(test_y)
  # print("[+] Test accuracy 1: {} ".format(accuracy))


  #Evaluate 2
  testAccuracy = model.evaluate(test_X, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(testAccuracy))
