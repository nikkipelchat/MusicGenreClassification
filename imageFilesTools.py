# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np

#Returns numpy image at size imageSizeX*imageSizeY
def getProcessedData(img,imageSizeX,imageSizeY):
    img = img.resize((imageSizeX,imageSizeY), resample=Image.ANTIALIAS)
    imgData = np.asarray(img, dtype=np.uint8).reshape(imageSizeX,imageSizeY,1)
    imgData = imgData/255.
    return imgData

#Returns numpy image at size imageSizeX*imageSizeY
def getImageData(filename,imageSizeX,imageSizeY):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSizeX, imageSizeY)
    return imgData