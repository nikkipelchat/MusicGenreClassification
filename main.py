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
parser.add_argument("mode", help="Trains or tests the CNN", nargs="*", choices=["train", "continue", "test", "confusionmatrix", "slice"])
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
  #Create or load new dataset
  train_x, train_y, validation_x, validation_y = getDataset(filesPerGenreMap, genres, mode="train")

  if args.resume and args.epochs:
    number_of_epochs = args.epochs
  else:
    number_of_epochs = nbEpoch

  #Define run id for graphs
  run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

  #Train the model
  print("[+] Training the model...")
  t0 = time.gmtime()
  model.fit(train_x, train_y, n_epoch=number_of_epochs, batch_size=batchSize, shuffle=True, validation_set=(validation_x, \
    validation_y), snapshot_epoch=True, show_metric=True, run_id=run_id)
  t1 = time.gmtime()
  seconds_to_train = time.mktime(t1) - time.mktime(t0)
  hours, minutes = divmod(seconds_to_train, 3600)
  print("[+]: Time to train: {} hours, {:.0f} minutes".format(hours, minutes/60))
  print("    Model trained!")

  #Save trained model
  print("[+] Saving the weights...")
  model.save('{}/model.tfl'.format(checkpointPath))
  print("[+] Weights saved!")

  print("[+] Test Neural Network")
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  print("[+] Loading weights...")
  model.load('{}/model.tfl'.format(checkpointPath))
  print("    Weights loaded!")
  #Evaluate 2
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))



# Load trained model, evaluate model and print test accuracy
if "test" in args.mode:
  print("[+] Test Neural Network")
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  print("[+] Loading weights...")
  model.load('{}/model.tfl'.format(checkpointPath))
  print("    Weights loaded!")
  #Evaluate 2
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))



if "confusionmatrix" in args.mode:
  #Create or load new dataset
  test_x, test_y = getDataset(filesPerGenreMap, genres, mode="test")

  #Predict and compare
  # actualClasses = test_y
  # predictedClasses = model.predict_label(test_x)
  # print("Length of test_x: {}".format(len(test_x)))
  # print("Predicted Classes: {}".format(predictedClasses))
  # print("Actual Classes: {}".format(actualClasses))

  #Load weights
  print("[+] Loading weights...")
  model.load('{}/model.tfl'.format(checkpointPath))
  print("    Weights loaded!")

  # Run the model on one example
  # prediction = model.predict([test_x[0]])
  # print("Prediction: %s" % str(prediction[0][:8]))

  #Evaluate 1
  # predictions = model.predict(test_x)#[test_x[0], test_x[1], test_x[2], test_x[3], test_x[4], test_x[5], test_x[6], \
  #   test_x[7], test_x[8], test_x[9], test_x[10]])
  # accuracy = 0
  # for prediction, actual in zip(predictions, test_y): #[test_y[0], test_y[1], test_y[2], test_y[3], test_y[4], test_y[5], \
  #   test_y[6], test_y[7], test_y[8], test_y[9], test_y[10]]):
  # 	predicted_class = np.argmax(prediction)
  # 	actual_class = np.argmax(actual)
  # 	print("Predicted: {} and actual: {}".format(predicted_class, actual_class))
  # 	if(predicted_class == actual_class):
  # 		accuracy+=1

  # accuracy = accuracy / len(test_y)
  # print("[+] Test accuracy 1: {} ".format(accuracy))

  # Confusion Matrix
  actual_classes = test_y[:1000]
  print("shape:", np.shape(actual_classes))
  predicted_classes = model.predict(test_x[:1000])
  # print("shape:", np.shape(actual_classes), " of ", actual_classes)
  # print("-------------------------")
  # print("shape:", np.shape(predicted_classes), " of ", predicted_classes)
  confusion_matrix = tf.confusion_matrix(labels=tf.argmax(actual_classes, 1), predictions=tf.argmax(predicted_classes, 1))
  with tf.Session():
    print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None))


  #Evaluate 2
  test_accuracy = model.evaluate(test_x, test_y)[0]
  print("[+] Test accuracy: {:.2%}".format(test_accuracy))
