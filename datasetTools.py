# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
from random import shuffle
import numpy as np
import pickle

from imageFilesTools import getImageData
from config import datasetPath
from config import slicesPath
from config import ignoreGenres
from config import spectrogramsPath

#Creates name of dataset from parameters
def getDatasetName(sliceXSize, sliceYSize):
  name = "{}".format(sliceXSize)
  name += "_{}".format(sliceYSize)
  return name

#Creates or loads dataset if it exists
#Mode = "train" or "test"
def getDataset(nbPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio, mode):
  print("[+] Dataset name: {}".format(getDatasetName(sliceXSize, sliceYSize)))
  if not os.path.isfile(datasetPath+"train_X_"+getDatasetName(sliceXSize, sliceYSize)+".p"):
    print("[+] Creating dataset with slices of size {}x{} per genre...".format(sliceXSize, sliceYSize))
    createDatasetFromSlices(nbPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio) 
  else:
    print("[+] Using existing dataset")
  
  return loadDataset(genres, sliceXSize, sliceYSize, mode)

#Loads dataset
#Mode = "train" or "test"
def loadDataset(genres, sliceXSize, sliceYSize, mode):
  #Load existing
  datasetName = getDatasetName(sliceXSize, sliceYSize)
  if mode == "train":
    print("[+] Loading training and validation datasets... ")
    train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath,datasetName), "rb" ))
    train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath,datasetName), "rb" ))
    validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
    validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
    print("    Training and validation datasets loaded!")
    return train_X, train_y, validation_X, validation_y

  else:
    print("[+] Loading testing dataset... ")
    test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
    test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
    print("    Testing dataset loaded!")
    return test_X, test_y

#Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, genres, sliceXSize, sliceYSize):
  #Create path for dataset if not existing
  if not os.path.exists(os.path.dirname(datasetPath)):
    try:
      os.makedirs(os.path.dirname(datasetPath))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise

  #SaveDataset
  print("[+] Saving dataset... ")
  datasetName = getDatasetName(sliceXSize, sliceYSize)
  pickle.dump(train_X, open("{}train_X_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  pickle.dump(train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  pickle.dump(validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  pickle.dump(test_X, open("{}test_X_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  pickle.dump(test_y, open("{}test_y_{}.p".format(datasetPath,datasetName), "wb" ), protocol=4)
  print("    Dataset saved!")

#Creates and save dataset from slices
def createDatasetFromSlices(nbPerGenreMap, genres, sliceXSize, sliceYSize, validationRatio, testRatio):
  validationData = []
  testingData = []
  trainingData = []

  for genre in genres:
    #Get slices in genre subfolder
    filenames = os.listdir(slicesPath+genre)
    filenames = [filename for filename in filenames if filename.endswith('.png')]

    # Number of files per genre map
    numberOfFilesPerGenre = nbPerGenreMap.get(genre, nbPerGenreMap.get('Default'))

    #If we're supposed to ignore genre, don't add to dataset
    if any(genre in s for s in ignoreGenres):
      print("-> Ignoring {}, {} slices".format(genre, len(filenames)))
      continue
    print("-> Adding {}, {} of {} slices".format(genre, numberOfFilesPerGenre, len(filenames)))

    filenames = filenames[:numberOfFilesPerGenre]

    #Split up files into train, test, and validate
    #Index of array to grab validation files (amount in config) of array
    arrayIndexForValidation = int(len(filenames)*validationRatio)
    #Index of array to grab test files (amount in config) of array
    arrayIndexForTest = int(len(filenames)*testRatio)
    #Index of array to grab the rest of array
    arrayIndexForTrain = int(len(filenames)*validationRatio)

    fileNameArraySplit = np.split(filenames, [arrayIndexForValidation, arrayIndexForTest + arrayIndexForValidation])

    validationFilenames = fileNameArraySplit[0]
    testFilenames = fileNameArraySplit[1]
    trainingFilenames = fileNameArraySplit[2]
    
    #Randomize file selection for this genre
    shuffle(validationFilenames)
    shuffle(testFilenames)
    shuffle(trainingFilenames)

    #Add data (X,y)
    for validationFilename in validationFilenames:
      imgData = getImageData(slicesPath+genre+"/"+validationFilename, sliceXSize, sliceYSize)
      label = [1. if genre == g else 0. for g in genres]
      validationData.append((imgData,label))

    for testFilename in testFilenames:
      imgData = getImageData(slicesPath+genre+"/"+testFilename, sliceXSize, sliceYSize)
      label = [1. if genre == g else 0. for g in genres]
      testingData.append((imgData,label))

    for trainingFilename in trainingFilenames:
      imgData = getImageData(slicesPath+genre+"/"+trainingFilename, sliceXSize, sliceYSize)
      label = [1. if genre == g else 0. for g in genres]
      trainingData.append((imgData,label))

#Shuffle data
shuffle(validationData)
shuffle(testingData)
shuffle(trainingData)
    
print("----------------")
print("Total dataset {}".format(len(trainingData) + len(validationData) + len(testingData)))
print("Split up into Training: {};  Validation: {};  Test: {};".format(len(trainingData), len(validationData), len(testingData)))

#Extract X and y
validate_X,validate_Y = zip(*validationData)
test_X,test_Y = zip(*testingData)
train_X,train_Y = zip(*trainingData)

#Prepare for Tflearn
train_X = np.array(train_X).reshape([-1, sliceXSize, sliceYSize, 1])
train_Y = np.array(train_Y)
validation_X = np.array(validate_X).reshape([-1, sliceXSize, sliceYSize, 1])
validation_Y = np.array(validate_Y)
test_X = np.array(test_X).reshape([-1, sliceXSize, sliceYSize, 1])
test_Y = np.array(test_Y)
print("    Dataset created!")
    
#Save
saveDataset(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, genres, sliceXSize, sliceYSize)

return train_X, train_Y, validation_X, validation_Y, test_X, test_Y
