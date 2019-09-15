'''Open image files and return data within the image to add to dataset'''
from PIL import Image
import numpy as np


def getProcessedData(img, imageSizeX, imageSizeY):
  '''Returns numpy image at size imageSizeX*imageSizeY'''
  img = img.resize((imageSizeX, imageSizeY), resample=Image.ANTIALIAS)
  imgData = np.asarray(img, dtype=np.uint8).reshape(imageSizeX, imageSizeY, 1)
  imgData = imgData/255.
  return imgData


def getImageData(filename, imageSizeX, imageSizeY):
  '''Returns numpy image at size imageSizeX*imageSizeY'''
  img = Image.open(filename)
  imgData = getProcessedData(img, imageSizeX, imageSizeY)
  return imgData
