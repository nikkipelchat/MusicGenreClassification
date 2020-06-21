'''All configuration values'''
# pylint: disable=invalid-name
# Define paths for files
spectrogramsPath = "Data/Spectrograms/"
linearSpectrogramsPath = "Data/Spectrograms/Linear/"
melSpectrogramsPath = "Data/Spectrograms/Mel/"
logSpectrogramsPath = "Data/Spectrograms/Log/"

slicesPath = "Data/Slices/"
datasetPath = "Data/Dataset/"
checkpointPath = "Checkpoints/"
rawDataPath = "Data/Raw/"

# Spectrogram resolution
pixelPerSecond = 50

# Slice parameters
sliceXSize = 128
sliceYSize = 128
sliceZSize = 1

# Dataset parameters
# filesPerGenreMap = { # uneven for cody's dataset to replicate mine
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
filesPerGenreMap = { # even slicing for cody's dataset
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
# filesPerGenreMap = { # for GTZAN
#    'Blues': 40000,
#    'Classical': 40000,
#    'Country': 40000,
#    'Disco': 40000,
#    'HipHop': 40000,
#    'Jazz': 40000,
#    'Metal': 40000,
#    'Pop': 40000,
#    'RBSoul': 40000,
#    'Rock': 40000,
# }
ignoreGenres = ['Other', 'Soundtrack']
validationRatio = 0.3
testRatio = 0.1

# Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 20
