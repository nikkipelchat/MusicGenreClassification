'''Open image files and return data within the image to add to dataset'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def getProcessedData(img, imageSizeX, imageSizeY, imageSizeZ):
  '''Returns numpy image at size X*Y*Z'''
  imgData = np.asarray(img, dtype=np.uint8).reshape(imageSizeX, imageSizeY, imageSizeZ)
  imgData = imgData/255.
  return imgData


def getImageData(filename, imageSizeX, imageSizeY, imageSizeZ):
  '''Returns numpy image at size X*Y*Z'''
  conversionType = 'L' # 8-bit pixels, black and white
  if (imageSizeZ == 3):
    conversionType = 'RGB' # 3x8-bit pixels, true color
  elif (imageSizeZ == 4):
    conversionType = 'RGBA' # 4x8-bit pixels, true color with transparency mask
  img = Image.open(filename).convert(conversionType)
  imgData = getProcessedData(img, imageSizeX, imageSizeY, imageSizeZ)
  return imgData
