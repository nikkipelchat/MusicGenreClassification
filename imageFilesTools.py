'''Open image files and return data within the image to add to dataset'''
import os
from PIL import Image
import numpy as np


def getProcessedData(img, imageSizeX, imageSizeY, imageSizeZ):
  '''Returns numpy image at size X*Y*Z'''
  imgData = np.asarray(img, dtype=np.uint8).reshape(imageSizeX, imageSizeY, imageSizeZ)
  imgData = imgData/255.
  return imgData


def getImageData(filename, imageSizeX, imageSizeY, imageSizeZ):
  '''Returns numpy image at size X*Y*Z'''
  try:
    conversionType = 'L' # 8-bit pixels, black and white
    if imageSizeZ == 3:
      conversionType = 'RGB' # 3x8-bit pixels, true color
    elif imageSizeZ == 4:
      conversionType = 'RGBA' # 4x8-bit pixels, true color with transparency mask
    img = Image.open(filename).convert(conversionType)
    imgData = getProcessedData(img, imageSizeX, imageSizeY, imageSizeZ)
    return imgData
  except KeyboardInterrupt:
    raise
  except: # pylint: disable=W0702
    print("Couldn't load image: {}".format(filename))
    raise


def createFolder(folderName):
  '''Create path if not existing'''
  if not os.path.exists(os.path.dirname(folderName)):
    try:
      os.makedirs(os.path.dirname(folderName))
    except OSError as exc: # Guard against race condition
      # pylint: disable=undefined-variable
      if exc.errno != errno.EEXIST:
        raise
