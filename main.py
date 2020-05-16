'''Main python script to run'''
import random
import string
import os
import time
import sys
import argparse
import tensorflow as tf
import numpy as np

from model import createModelUsingTensorflow
from datasetTools import getDataset
from config import slicesPath
from config import checkpointPath
from config import batchSize
from config import filesPerGenreMap
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceXSize, sliceYSize, sliceZSize
from config import ignoreGenres

from songToData import createSlicesFromAudio

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs="*", choices=["train", "continue", "test", "confusionmatrix", "vote", "slice"])
parser.add_argument("--resume", help="The version to continue training from", required=False, default=False)
parser.add_argument("--epochs", help="The number of epochs to finish training from", type=int, required=False, default=False)
args = parser.parse_args()
print("args", args)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only log errors

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Batch size: {}".format(batchSize))

if "slice" in args.mode:
  createSlicesFromAudio()
  sys.exit()

# List genres
all_genres = os.listdir(slicesPath)
all_genres = [filename for filename in all_genres if os.path.isdir(slicesPath+filename)]
set_of_all_genres = set(all_genres)
set_of_genres_to_ignore = set(ignoreGenres)
# Genres to use
genres = set_of_all_genres - set_of_genres_to_ignore
print("| Genres: {}".format(genres))
number_of_classes = len(genres)

print("| Number of classes: {}".format(number_of_classes))
print("| Slices per genre map: {}".format(filesPerGenreMap))
print("| Slice size: {}x{}x{}".format(sliceXSize, sliceYSize, sliceZSize))
print("--------------------------")

# Create model or resume training
model = createModelUsingTensorflow(number_of_classes, sliceXSize, sliceYSize, sliceZSize, args)

# Get dataset, train the model created
if "train" in args.mode:
  # Create or load new dataset
  train_x, train_y, validation_x, validation_y = getDataset(filesPerGenreMap, genres, mode="train")

  if args.resume and args.epochs:
    number_of_epochs = args.epochs
  else:
    number_of_epochs = nbEpoch

  # Define run id for graphs
  run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

  # Train the model
  print("[+] Training the model...")
  t0 = time.gmtime()
  model.fit(train_x, train_y, n_epoch=number_of_epochs, batch_size=batchSize, shuffle=True, validation_set=(validation_x, \
    validation_y), snapshot_epoch=True, show_metric=True, run_id=run_id)
  t1 = time.gmtime()
  seconds_to_train = time.mktime(t1) - time.mktime(t0)
  hours, minutes = divmod(seconds_to_train, 3600)
  print("[+]: Time to train: {} hours, {:.0f} minutes".format(hours, minutes/60))
  print("    Model trained!")

  # Save trained model
  print("[+] Saving the weights...")
  model.save('{}/model.tfl'.format(checkpointPath))
  print("[+] Weights saved!")

  print("[+] Test Neural Network")
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  print("[+] Loading weights...")
  model.load('{}/model.tfl'.format(checkpointPath))
  print("    Weights loaded!")

  # Evaluate
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))


# Load trained model, evaluate model and print test accuracy
if "test" in args.mode:
  print("[+] Test Neural Network")
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  model.load('{}/model.tfl'.format(checkpointPath))

  # Evaluate
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))



if "confusionmatrix" in args.mode:
  print("[+] Create a Confusion Matrix")
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  # Load weights
  model.load('{}/model.tfl'.format(checkpointPath))

  # Confusion Matrix
  chunkSize = 1000
  actual_classes = test_y
  predictionsArray = np.empty([0, number_of_classes])
  for i in range(0, len(test_x), chunkSize):
    testXChunk = test_x[i:i+chunkSize]
    predictionChunk = model.predict(testXChunk)
    predictionsArray = np.concatenate((predictionsArray, predictionChunk))
  
  labels = tf.argmax(actual_classes, 1)
  predictions = tf.argmax(predictionsArray, 1)

  confusion_matrix = tf.confusion_matrix(labels=labels, predictions=predictions)
  with tf.Session():
    print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None))

  # Evaluate
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))


if "vote" in args.mode:
  # Create or load new dataset
  test_x, test_y, song_titles_for_votes = getDataset(filesPerGenreMap, genres, mode="vote")

  # Load weights
  model.load('{}/model.tfl'.format(checkpointPath))

  songPredictionTotals = {}
  songActualClassDict = {}
  accuracy = 0
  voteAccuracy = 0

  chunkSize = 1000

  # Evaluate accuracy and create dictionary of how often slices within 1 song are accurate
  for i in range(0, len(song_titles_for_votes), chunkSize):
    testXChunk = test_x[i:i+chunkSize]
    testYChunk = test_y[i:i+chunkSize]
    songTitlesForVotesChunk = song_titles_for_votes[i:i+chunkSize]

    predictionChunk = model.predict(testXChunk)

    for prediction, actual, songName in zip(predictionChunk, testYChunk, songTitlesForVotesChunk):
      predicted_class_for_slice = np.argmax(prediction)
      actual_class_for_slice = np.argmax(actual)

      if not songPredictionTotals.get(songName):
        songPredictionTotals[songName] = {}
      if not songPredictionTotals[songName].get(predicted_class_for_slice):
        songPredictionTotals[songName][predicted_class_for_slice] = 0 # to keep the total number of predictions for this class 

      songPredictionTotals[songName][predicted_class_for_slice] += 1
      songActualClassDict[songName] = actual_class_for_slice

      if (predicted_class_for_slice == actual_class_for_slice):
        accuracy += 1

  # Extract vote accuracy
  for songName, songNameItems in songPredictionTotals.items():
    most_predicted_class_for_song = max(songNameItems, key=lambda key: songNameItems[key])
    actual_class_for_song = songActualClassDict[songName]
    if (most_predicted_class_for_song == actual_class_for_song):
      voteAccuracy += 1

  accuracy = accuracy / len(test_y)
  print("[+] Calculated Test accuracy: {:.2%}".format(accuracy))

  voteAccuracy = voteAccuracy / len(songPredictionTotals)
  print("[+] Calculated Voted Test accuracy: {:.2%}".format(voteAccuracy))

  # Evaluate
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Model Calculated Test accuracy: {:.2%}".format(test_accuracy))
