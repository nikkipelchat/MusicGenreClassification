# Import Pillow:
from PIL import Image
import os.path

from config import spectrogramsPath, slicesPath

#Slices all spectrograms
def createSlicesFromSpectrograms(sliceXSize, sliceYSize):
	for filename in os.listdir(spectrogramsPath):
		if filename.endswith(".png"):
			sliceSpectrogram(filename, sliceXSize, sliceYSize)

#Creates slices from spectrogram
#TODO Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, sliceXSize, sliceYSize):
	genre = filename.split("_")[0] 	#Ex. Dubstep_19.png

	# Load the full spectrogram
	img = Image.open(spectrogramsPath+filename)

	#Compute approximate number of sizeX x sizeY samples
	width, height = img.size
	nbSamples = int(width/sliceXSize)
	width - sliceXSize

	#Create path if not existing
	slicePath = slicesPath+"{}/".format(genre);
	if not os.path.exists(os.path.dirname(slicePath)):
		try:
			os.makedirs(os.path.dirname(slicePath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	#For each sample
	for i in range(nbSamples):
		if i == 0:
			print("Creating {} slices for {}: ".format(nbSamples, filename))
		#Extract and save sizeX x sizeY sample
		startPixel = i*sliceXSize
		imgTmp = img.crop((startPixel, 1, startPixel + sliceXSize, sliceYSize + 1))
		imgTmp.save(slicesPath+"{}/{}_{}.png".format(genre,filename[:-4],i))

