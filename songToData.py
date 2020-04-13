'''Create spectrograms and slices from audio files'''
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

def createSpectrogramMelScale(filename, newFilename):
  '''Create a mel scale spectrogram'''
  # Set up graph
  plt.axis('off')

  try:
    # Create Spectrogram
    y, sr = librosa.load(rawDataPath+filename, sr=22050, mono=True, offset=20) # offset 20 trims 20 seconds off the start of the song
    ps = librosa.feature.melspectrogram(y=y, sr=sr, window='hamming')
    height, width = ps.shape
    heigthOfSpec = round(height/100, 2)
    widthOfSpec = round((width*1.16075)/100, 2)

    plt.figure(figsize=(widthOfSpec, heigthOfSpec))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

    # Display Spectrogram
    power = librosa.power_to_db(ps, ref=np.max)
    # to see amplitude it would be S**2 https://librosa.github.io/librosa/generated/librosa.core.amplitude_to_db.html
    amplitude = librosa.power_to_db(ps**2, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # binary mel
    # librosa.display.specshow(power, sr=sr, cmap='binary', x_axis='time', y_axis='mel')
    # plt.savefig('{}'.format(spectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)
    # librosa.display.specshow(amplitude, sr=sr, cmap='binary', x_axis='time', y_axis='mel')
    # plt.savefig('{}'.format(spectrogramsPathToMelAmplitude+newFilename), bbox_inches=None, pad_inches=0)

    # greys mel
    # librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), sr=sr, cmap='Greys', x_axis='time', y_axis='mel')
    # plt.savefig('{}'.format(spectrogramsPathToGreys+newFilename), bbox_inches=None, pad_inches=0)

    # colored mel
    librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.savefig('{}'.format(spectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)

    # binary log
    # librosa.display.specshow(power, sr=sr, cmap='binary', x_axis='time', y_axis='log')
    # plt.savefig('{}'.format(spectrogramsPathToLogPower+newFilename), bbox_inches=None, pad_inches=0)
    # librosa.display.specshow(amplitude, sr=sr, cmap='binary', x_axis='time', y_axis='log')
    # plt.savefig('{}'.format(spectrogramsPathToLogAmplitude+newFilename), bbox_inches=None, pad_inches=0)

    # binary linear
    # librosa.display.specshow(power, sr=sr, cmap='binary', x_axis='time')
    # plt.savefig('{}'.format(spectrogramsPathToLinearPower+newFilename), bbox_inches=None, pad_inches=0)
    # librosa.display.specshow(amplitude, sr=sr, cmap='binary', x_axis='time')
    # plt.savefig('{}'.format(spectrogramsPathToLinearAmplitude+newFilename), bbox_inches=None, pad_inches=0)

    # binary mfcc
    # librosa.display.specshow(mfccs, cmap='binary', x_axis='time', y_axis='linear')
    # plt.savefig('{}'.format(spectrogramsPathToMFCCLinear+newFilename), bbox_inches=None, pad_inches=0)
    # librosa.display.specshow(mfccs, cmap='binary', x_axis='time')
    # plt.savefig('{}'.format(spectrogramsPathToMFCC+newFilename), bbox_inches=None, pad_inches=0)

  except KeyboardInterrupt:
    raise
  except:
    print("Couldn't create spectrogram(s) for {}".format(filename))

  # Save Spectrogram
  plt.close()


def createMFCC(filename, newFilename):
  '''Create an MFCC spectrogram'''
  # Set up graph
  plt.axis('off')
  plt.figure(figsize=(80, 1.28))
  plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

  try:
    # Create Spectrogram
    y, sr = librosa.load(rawDataPath+filename, sr=22050, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Display Spectrogram
    librosa.display.specshow(mfccs, x_axis='time', cmap='binary')
    # librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
  except KeyboardInterrupt:
    raise
  except:
    print("Couldn't create MFCC for {}".format(filename))

  # Save Spectrogram
  plt.savefig('{}'.format(spectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)
  plt.close()


def createSpectrogramLibrosaSpectrogram(filename, newFilename):
  '''Create a spectrogram using Librosa Library'''
  # Set up graph
  plt.axis('off')
  plt.figure(figsize=(80, 1.28))
  plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

  try:
    # Create Spectrogram
    y, sr = librosa.load(rawDataPath+filename, sr=22050, mono=True)
    # D = librosa.stft(y)
    # S = np.abs(librosa.stft(y))
    # db_to_amplitude(S, )
    # magnitude, phase = librosa.magphase(D)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # S=librosa.power_to_db(S)
    # ps = librosa.feature.melspectrogram(y=y, sr=sr)

    # Display Spectrogram
    # librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max), sr=sr, cmap='gray_r', x_axis='time', y_axis='linear')
    # librosa.display.specshow(librosa.power_to_db(ps, ref=np.median), sr=sr, cmap='gray_r', x_axis='time', y_axis='linear')
    librosa.display.specshow(mfcc, cmap='gray_r', x_axis='time')

  except KeyboardInterrupt:
    print("Keyboard Interruption")
    raise
  except:
    print("Couldn't create spectrogram for {}".format(filename))

  # Save Spectrogram
  plt.savefig('{}'.format(spectrogramsPath+newFilename), bbox_inches=None, pad_inches=0)
  plt.close()


def createSpectrogramSox(filename, newFilename):
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
    pixelPerSecond, spectrogramsPath + newFilename)
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
  createFolder(spectrogramsPath)

  # Rename files according to genre
  for index, filename in enumerate(files):
    # Rename file before finding genre
    # Strip out any special characters
    os.rename(rawDataPath+filename, rawDataPath+re.sub('[^A-Za-z0-9. ]+', '', filename))
    filename = re.sub('[^A-Za-z0-9. ]+', '', filename)

    print("Creating spectrogram for file {}/{}...  {}".format(index+1, nbFiles, filename))

    fileGenre = getGenre(rawDataPath + filename)
    if fileGenre == 'Other':
      print("Genre was converted into 'other' for file: ", filename)
    genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
    fileID = genresID[fileGenre]
    newFilename = fileGenre+"_"+str(fileID) # if fileGenre is byte then do this
    createSpectrogramMelScale(filename, newFilename)
    # createMFCC(filename, newFilename+"_mfcc")
    # createSpectrogramLibrosaSpectrogram(filename, newFilename+"_librosa_spectrogram")
    # createSpectrogramSox(filename, newFilename)


def createSlicesFromAudio():
  '''Whole pipeline .mp3 -> .png slices'''
  print("Creating spectrograms...")
  createSpectrogramsFromAudio()
  print("Spectrograms created!")

  print("Creating slices for images...")
  createSlicesFromSpectrograms(spectrogramsPath, slicesPath, sliceXSize, sliceYSize)
  print("Slices created!")
