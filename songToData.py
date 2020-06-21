'''Create spectrograms and slices from audio files'''
# pylint: disable=C0103
from subprocess import Popen, PIPE, STDOUT
import os
import sys
import re
import warnings
import eyed3

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from imageFilesTools import createFolder
from config import rawDataPath
from config import spectrogramsPath
from config import melSpectrogramsPath, linearSpectrogramsPath, logSpectrogramsPath
from config import slicesPath
from config import pixelPerSecond
from config import sliceXSize, sliceYSize

warnings.filterwarnings('ignore')

print(sys.executable)
print(sys.version)

# Define
current_path = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")

def createMelSpectrogramLibrosa(filename, newFilename):
  '''Create a mel scale spectrogram'''
  # IF statement allows creating spectrograms to resume if the task failed
  if os.path.exists(melSpectrogramsPath+newFilename+'.png'):
    print("Spectrogram ", melSpectrogramsPath+newFilename, " already exists.")
  else:
    # Set up graph
    plt.axis('off')

    try:
      # Create Spectrogram
      y, sr = librosa.load(rawDataPath+filename, sr=44100, mono=True, offset=20) # offset 20 trims 20 seconds off the start of the song
      # y, sr = librosa.load(rawDataPath+filename, sr=22050, mono=True, offset=20) # GTZAN
      ps = librosa.feature.melspectrogram(y=y, sr=sr, window='hamming')
      # Same thing as line above
      #   D = np.abs(librosa.stft(y)) # power spectrogram
      #   ps = librosa.feature.melspectrogram(S=D, sr=sr, window='hamming')
      power = librosa.power_to_db(ps, ref=np.max)

      height, width = power.shape # height = 128, width depends on song length
      heightOfSpec = round(height/100, 2)
      widthOfSpec = round((width*0.58045)/100, 2) # Personal music
      # widthOfSpec = round((width*1.16075)/100, 2) # GTZAN

      plt.figure(figsize=(widthOfSpec, heightOfSpec))
      plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

      # binary mel
      librosa.display.specshow(power, sr=sr, cmap='binary', x_axis='time', y_axis='mel', fmax=sr/2) # need fmax for mel scale
      plt.savefig('{}'.format(melSpectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)

    except KeyboardInterrupt:
      raise
    except: # pylint: disable=W0702
      print("Couldn't create spectrogram(s) for {}".format(filename))
      print("Error: {}".format(sys.exc_info()[0]))

    # Save Spectrogram
    plt.close()


def createLinearOrLogSpectrogramLibrosa(filename, newFilename):
  '''Create a linear and logarithmic scale spectrogram'''
  # IF statement allows creating spectrograms to resume if the task failed
  if os.path.exists(logSpectrogramsPath+newFilename+'.png'):
    print("Spectrogram ", logSpectrogramsPath+newFilename, " already exists.")
  else:
    # Set up graph
    plt.axis('off')

    try:
      # Create Spectrogram
      y, sr = librosa.load(rawDataPath+filename, sr=44100, mono=True, offset=20) # offset 20 -> trims 20 seconds off the start of the song
      # y, sr = librosa.load(rawDataPath+filename, sr=22050, mono=True, offset=20) # GTZAN
      D = np.abs(librosa.stft(y)) # power spectrogram
      S = librosa.power_to_db(D**2, ref=np.max) # spectrogram for linear or logithmic

      height, width = S.shape # height = 1025, width depends on song length
      heightOfSpec = round((height*0.12585)/100, 2)
      widthOfSpec = round((width*0.58045)/100, 2) # Personal music
      # widthOfSpec = round((width*1.16075)/100, 2) # GTZAN

      plt.figure(figsize=(widthOfSpec, heightOfSpec))
      plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

      # binary linear
      # librosa.display.specshow(S, sr=sr, cmap='binary', x_axis='time', y_axis='linear') # need to load at 44100
      # plt.savefig('{}'.format(linearSpectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)

      # binary logarithmic
      librosa.display.specshow(S, sr=sr, cmap='binary', x_axis='time', y_axis='log') # need to load at 44100
      plt.savefig('{}'.format(logSpectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)

    except KeyboardInterrupt:
      raise
    except: # pylint: disable=W0702
      print("Couldn't create spectrogram(s) for {}".format(filename))
      print("Error: {}".format(sys.exc_info()[0]))

    # Save Spectrogram
    plt.close()


def createMFCCPlotLibrosa(filename, newFilename):
  '''Create a plot of MFCC co-efficents'''
  # IF statement allows creating spectrograms to resume if the task failed
  if os.path.exists(spectrogramsPath+newFilename+'.png'):
    print("Spectrogram ", spectrogramsPath+newFilename, " already exists.")
  else:
    # Set up graph
    plt.axis('off')

    try:
      # Create Spectrogram
      y, sr = librosa.load(rawDataPath+filename, sr=44100, mono=True, offset=20) # offset 20 trims 20 seconds off the start of the song
      mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
      height, width = mfccs.shape
      heightOfSpec = round(height/100, 2)
      widthOfSpec = round((width*0.58045)/100, 2)

      plt.figure(figsize=(widthOfSpec, heightOfSpec)) # this might not be the width/height we want, it was from creating a spectrogram
      plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

      # binary mfcc
      librosa.display.specshow(mfccs, cmap='binary', x_axis='time')
      plt.savefig('{}'.format(spectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)

    except KeyboardInterrupt:
      raise
    except: # pylint: disable=W0702
      print("Couldn't create spectrogram(s) for {}".format(filename))
      print("Error: {}".format(sys.exc_info()[0]))

    # Save Spectrogram
    plt.close()


def createLinearSpectrogramSox(filename, newFilename):
  '''Create spectrogram from mp3 files'''
  # Create temporary mono track if needed
  if isMono(rawDataPath + filename):
    command = "cp '{}' '/tmp/{}.mp3'".format(rawDataPath + filename, newFilename)
  else:
    command = 'sox "{}" "/tmp/{}.mp3" remix 1,2'.format(rawDataPath + filename, newFilename)
  process = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=current_path)
  # pylint: disable=unused-variable
  output, errors = process.communicate()
  if errors:
    print(errors)

  # Create spectrogram
  filename.replace(".mp3", "")
  # trim 20 will cut 20 seconds off front of song
  command = 'sox "/tmp/{}.mp3" -n trim 20 spectrogram -w Hamming -Y 200 -X {} -m -r -o "{}.png"'.format(newFilename, \
    pixelPerSecond, spectrogramsPath + newFilename) # -r removes axes
  process = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=current_path)
  # pylint: disable=unused-variable
  output, errors = process.communicate()
  if errors:
    print(errors)

  # Remove tmp mono track
  os.remove("/tmp/{}.mp3".format(newFilename))


def createSpectrogramsFromAudio():
  '''Creates .png whole spectrograms from mp3 files'''
  genresID = dict()
  files = os.listdir(rawDataPath)
  files = [file for file in files if file.endswith(".mp3")]
  nbFiles = len(files)

  # Create path if not existing
  createFolder(linearSpectrogramsPath)
  createFolder(melSpectrogramsPath)
  createFolder(logSpectrogramsPath)

  # Rename files according to genre
  for index, filename in enumerate(files):
    # Rename file before finding genre
    # Strip out any special characters
    os.rename(rawDataPath+filename, rawDataPath+re.sub('[^A-Za-z0-9. ]+', '', filename))
    filename = re.sub('[^A-Za-z0-9. ]+', '', filename)

    print("Creating spectrogram(s) for file {}/{}...  {}".format(index+1, nbFiles, filename))

    fileGenre = getGenre(rawDataPath + filename)
    if fileGenre == 'Other':
      print("Genre was converted into 'other' for file: ", filename)
    genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
    fileID = genresID[fileGenre]
    newFilename = fileGenre+"_"+str(fileID) # if fileGenre is byte then do this
    createMelSpectrogramLibrosa(filename, newFilename)
    # createMFCCPlotLibrosa(filename, newFilename)
    createLinearOrLogSpectrogramLibrosa(filename, newFilename)
    # createLinearSpectrogramSox(filename, newFilename)


def createSlicesFromAudio():
  '''Whole pipeline .mp3 -> .png slices'''
  print("Creating spectrograms...")
  createSpectrogramsFromAudio()
  print("Spectrograms created!")

  print("Creating slices for images...")
  createSlicesFromSpectrograms(spectrogramsPath, slicesPath, sliceXSize, sliceYSize)
  print("Slices created!")
