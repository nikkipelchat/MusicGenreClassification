#Define paths for files
spectrogramsPath = "Data/Spectrograms/"
slicesPath = "Data/Slices/"
datasetPath = "Data/Dataset/"
rawDataPath = "Data/Raw/"

#Spectrogram resolution
pixelPerSecond = 50

#Slice parameters
sliceSize = 128
sliceXSize = 128
sliceYSize = 128

#Dataset parameters
# filesPerGenreMap = {
#   'HipHop': 32953,
#   'Rap': 2785,
#   'Electronic': 13574,
#   'Dance': 0,
#   'Blues': 14852,
#   'Rock': 40000, 
#   'Pop': 9294,
#   'RBSoul': 1931,
#   'Alternative': 4946,
#   'Country': 700,
#   'Classical': 7372,
#   'Default': 0,
# }
filesPerGenreMap = {
  'HipHop': 20000,
  'Rap': 20000,
  'Electronic': 20000,
  'Dance': 20000,
  'Blues': 20000,
  'Rock': 20000,
  'Pop': 20000,
  'RBSoul': 20000,
  'Alternative': 20000,
  'Country': 20000,
  'Classical': 20000,
  'Default': 50,
}
ignoreGenres = ['Other', 'Soundtrack']
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 15