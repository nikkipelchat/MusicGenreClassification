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
 'HipHop': 15000,
 'Rap': 15000,
 'Electronic': 15000,
 'Dance': 15000,
 'Blues': 15000,
 'Rock': 15000,
 'Pop': 15000,
 'RBSoul': 15000,
 'Alternative': 15000,
 'Country': 15000,
 'Classical': 15000,
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
learningRate = 0.000147
nbEpoch = 20
