'''Slice up all spectrograms which are the length of the full song'''
import os.path
from PIL import Image
from imageFilesTools import createFolder


def createSlicesFromSpectrograms(spectrogramsPath, slicesPath, sliceXSize, sliceYSize):
  '''Slices all spectrograms'''
  for filename in os.listdir(spectrogramsPath):
    if filename.endswith(".png"):
      try:
        sliceSpectrogram(filename, spectrogramsPath, slicesPath, sliceXSize, sliceYSize)
      except KeyboardInterrupt:
        raise
      except:
        print("Couldn't create slices for {}".format(filename))


def sliceSpectrogram(filename, spectrogramPath, slicesPath, sliceXSize, sliceYSize):
  '''Creates slices from spectrogram'''
  genre = filename.split("_")[0] 	#Ex. Dubstep_19.png

  # Load the full spectrogram
  img = Image.open(spectrogramPath+filename)

  # Compute approximate number of sizeX x sizeY samples
  # pylint: disable=unused-variable
  width, height = img.size
  expectedNumberOfSamples = int(width/sliceXSize)
  actualNumberOfSamples = 0

  # Create path if not existing
  createFolder(slicesPath+"{}/".format(genre))

  # For each sample
  for i in range(expectedNumberOfSamples):
    # Extract and save sizeX x sizeY sample
    startPixel = i*sliceXSize
    contrastDifference = getContrastDifference(img, startPixel, sliceXSize, sliceYSize)

    if contrastDifference > 30:
      imgTmp = img.crop((startPixel, 1, startPixel + sliceXSize, sliceYSize + 1))
      # imgTmp.save(slicesPath+"{}/{}_{}_{}.png".format(genre,filename[:-4],i,'v2'))
      imgTmp.save(slicesPath+"{}/{}_{}.png".format(genre, filename[:-4], i))
      actualNumberOfSamples = actualNumberOfSamples + 1

  print("Created {}/{} slices for {}: ".format(actualNumberOfSamples, expectedNumberOfSamples, filename))


def getContrastDifference(img, startPixel, sliceXSize, sliceYSize):
  ''' determine difference in contrast of the image to see if its almost all white or all black '''
  imgTmp = img.crop((startPixel, 1, startPixel + sliceXSize, sliceYSize))
  extremaLow, extremaHigh = imgTmp.convert("L").getextrema()
  contrastDifference = extremaHigh - extremaLow
  if contrastDifference < 30 and extremaLow != 0 and extremaHigh != 255:
    print("    Slice was ignored but wasn't black or white.  (extremaLow, extremaHigh) -> ({}, {})".format(extremaLow, extremaHigh))
  return contrastDifference
