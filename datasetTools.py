'''Creates a Dataset and holds all code to do with a dataset'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from random import shuffle
import numpy as np

from imageFilesTools import getImageData, createFolder
from config import datasetPath
from config import slicesPath
from config import ignoreGenres
from config import validationRatio, testRatio
from config import sliceXSize, sliceYSize, sliceZSize

def getDatasetName():
  '''Creates name of dataset from parameters'''
  name = "{}".format(sliceXSize)
  name += "_{}".format(sliceYSize)
  name += "_{}".format(sliceZSize)
  return name


def getDataset(nbPerGenreMap, genres, mode):
  '''Creates or loads dataset if it exists. Mode = `train` or `test`'''
  print("[+] Dataset name: {}".format(getDatasetName()))
  if not os.path.isfile(datasetPath+"trainX_"+getDatasetName()+".p"):
    print("[+] Creating dataset with slices of size {}x{}x{} per genre...".format(sliceXSize, sliceYSize, sliceZSize))
    createDataset(nbPerGenreMap, genres)
  else:
    print("[+] Using existing dataset")

  return loadDataset(mode)


def loadDataset(mode):
  '''Loads datset. Mode = `train` or `test`'''
  #Load existing
  datasetName = getDatasetName()
  if mode == "train":
    print("[+] Loading training and validation datasets... ")
    trainX = pickle.load(open("{}trainX_{}.p".format(datasetPath, datasetName), "rb"))
    trainY = pickle.load(open("{}trainY_{}.p".format(datasetPath, datasetName), "rb"))
    validationX = pickle.load(open("{}validationX_{}.p".format(datasetPath, datasetName), "rb"))
    validationY = pickle.load(open("{}validationY_{}.p".format(datasetPath, datasetName), "rb"))
    print("    Training and validation datasets loaded!")
    return trainX, trainY, validationX, validationY

  # mode == "test"
  print("[+] Loading testing dataset... ")
  testX = pickle.load(open("{}testX_{}.p".format(datasetPath, datasetName), "rb"))
  testY = pickle.load(open("{}testY_{}.p".format(datasetPath, datasetName), "rb"))
  print("    Testing dataset loaded!")
  return testX, testY


def saveDataset(trainX, trainY, validationX, validationY, testX, testY):
  '''Saves dataset'''
  createFolder(datasetPath)

  # SaveDataset
  print("[+] Saving dataset... ")
  datasetName = getDatasetName()
  pickle.dump(trainX, open("{}trainX_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  pickle.dump(trainY, open("{}trainY_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  pickle.dump(validationX, open("{}validationX_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  pickle.dump(validationY, open("{}validationY_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  pickle.dump(testX, open("{}testX_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  pickle.dump(testY, open("{}testY_{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
  print("    Dataset saved!")


def splitFilesIntoTrainingValidationAndTestArrays(filenames):
  '''Split up the data into 3 arrays'''
  # Split up files into train, test, and validate
  # Index of array to grab validation files (amount in config) of array
  arrayIndexForValidation = int(len(filenames)*validationRatio)
  # Index of array to grab test files (amount in config) of array
  arrayIndexForTest = int(len(filenames)*testRatio)
  # Index of array to grab the rest of array
  # arrayIndexForTrain = int(len(filenames)*validationRatio)

  fileNameArraySplit = np.split(filenames, [arrayIndexForValidation, arrayIndexForTest + arrayIndexForValidation])

  validationFilenames = fileNameArraySplit[0]
  testFilenames = fileNameArraySplit[1]
  trainingFilenames = fileNameArraySplit[2]

  # Randomize file selection for this genre
  shuffle(trainingFilenames)
  shuffle(validationFilenames)
  shuffle(testFilenames)

  return trainingFilenames, validationFilenames, testFilenames


def addDataArraysToDataset(trainingFilenames, validationFilenames, testFilenames, \
  trainingData, validationData, testingData, genre, genres):
  '''Take arrays of file names and put them into correct section of dataset'''
  # Add data (X,Y)
  for validationFilename in validationFilenames:
    imgData = getImageData(slicesPath+genre + "/" + validationFilename, sliceXSize, sliceYSize, sliceZSize)
    label = [1. if genre == g else 0. for g in genres]
    validationData.append((imgData, label))

  for testFilename in testFilenames:
    imgData = getImageData(slicesPath+genre + "/" + testFilename, sliceXSize, sliceYSize, sliceZSize)
    label = [1. if genre == g else 0. for g in genres]
    testingData.append((imgData, label))

  for trainingFilename in trainingFilenames:
    imgData = getImageData(slicesPath+genre + "/" + trainingFilename, sliceXSize, sliceYSize, sliceZSize)
    label = [1. if genre == g else 0. for g in genres]
    trainingData.append((imgData, label))


def getDataForDataset(nbPerGenreMap, genres):
  '''For each genre it will grab all slices and split them up amongst training, validation, and test datasets'''
  trainingData = []
  validationData = []
  testingData = []

  for genre in genres:
    #Get slices in genre subfolder
    filenames = os.listdir(slicesPath+genre)
    filenames = [filename for filename in filenames if filename.endswith('.png')]

    # Number of files per genre map
    numberOfFilesPerGenre = nbPerGenreMap.get(genre, nbPerGenreMap.get('Default'))

    # If we're supposed to ignore genre, don't add to dataset
    if any(genre in s for s in ignoreGenres):
      print("-> Ignoring {}, {} slices".format(genre, len(filenames)))
      continue
    if numberOfFilesPerGenre > len(filenames):
      print("-> Adding {}, {} of {} slices".format(genre, len(filenames), len(filenames)))
    else:
      print("-> Adding {}, {} of {} slices".format(genre, numberOfFilesPerGenre, len(filenames)))

    filenames = filenames[:numberOfFilesPerGenre]

    trainingFilenames, validationFilenames, testFilenames = splitFilesIntoTrainingValidationAndTestArrays(filenames)

    addDataArraysToDataset(trainingFilenames, validationFilenames, testFilenames, \
      trainingData, validationData, testingData, genre, genres)

  return trainingData, validationData, testingData


def createDataset(nbPerGenreMap, genres):
  '''Creates and save dataset from slices'''
  trainingData, validationData, testingData = getDataForDataset(nbPerGenreMap, genres)

  #Shuffle data
  shuffle(validationData)
  shuffle(testingData)
  shuffle(trainingData)

  print("----------------")
  print("Total dataset {}".format(len(trainingData) + len(validationData) + len(testingData)))
  print("Split up into Training: {};  Validation: {};  Test: {};".format(len(trainingData), len(validationData), len(testingData)))

  #Extract X and y
  validateX, validateY = zip(*validationData)
  testX, testY = zip(*testingData)
  trainX, trainY = zip(*trainingData)

  #Prepare for Tflearn
  trainX = np.array(trainX).reshape([-1, sliceXSize, sliceYSize, sliceZSize]) # images/data
  trainY = np.array(trainY) # labels
  validationX = np.array(validateX).reshape([-1, sliceXSize, sliceYSize, sliceZSize]) # images/data
  validationY = np.array(validateY) # labels
  testX = np.array(testX).reshape([-1, sliceXSize, sliceYSize, sliceZSize]) # images/data
  testY = np.array(testY) # labels
  print("    Dataset created!")

  #Save
  saveDataset(trainX, trainY, validationX, validationY, testX, testY)

  return trainX, trainY, validationX, validationY, testX, testY
