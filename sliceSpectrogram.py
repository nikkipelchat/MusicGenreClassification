'''Slice up all spectrograms which are the length of the full song'''
import os.path
from PIL import Image


def createSlicesFromSpectrograms(spectrogramsPath, slicesPath, sliceXSize, sliceYSize):
  '''Slices all spectrograms'''
  for filename in os.listdir(spectrogramsPath):
    if filename.endswith(".png"):
      sliceSpectrogram(filename, spectrogramsPath, slicesPath, sliceXSize, sliceYSize)


# Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, spectrogramPath, slicesPath, sliceXSize, sliceYSize):
  '''Creates slices from spectrogram'''
  genre = filename.split("_")[0] 	#Ex. Dubstep_19.png

  # Load the full spectrogram
  img = Image.open(spectrogramPath+filename)

  # Compute approximate number of sizeX x sizeY samples
  # pylint: disable=unused-variable
  width, height = img.size
  nbSamples = int(width/sliceXSize)

  # Create path if not existing
  slicePath = slicesPath+"{}/".format(genre)
  if not os.path.exists(os.path.dirname(slicePath)):
    try:
      os.makedirs(os.path.dirname(slicePath))
    except OSError as exc: # Guard against race condition
      # pylint: disable=undefined-variable
      if exc.errno != errno.EEXIST:
        raise

  # For each sample
  for i in range(nbSamples):
    if i == 0:
      print("Creating {} slices for {}: ".format(nbSamples, filename))
    # Extract and save sizeX x sizeY sample
    startPixel = i*sliceXSize
    imgTmp = img.crop((startPixel, 1, startPixel + sliceXSize, sliceYSize + 1))
    # imgTmp.save(slicesPath+"{}/{}_{}_{}.png".format(genre,filename[:-4],i,'v2'))
    imgTmp.save(slicesPath+"{}/{}_{}.png".format(genre, filename[:-4], i))
