'''Create spectrograms and slices from audio files'''
from subprocess import Popen, PIPE, STDOUT
import os
import sys
import re
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond
from config import sliceXSize, sliceYSize

print(sys.executable)
print(sys.version)

# Define
current_path = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")


def createSpectrogram(filename, newFilename):
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
  # print("Made mono file")

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
  # print("Made sprectrogram")

  # Remove tmp mono track
  os.remove("/tmp/{}.mp3".format(newFilename))


def createSpectrogramsFromAudio():
  '''Creates .png whole spectrograms from mp3 files'''
  genresID = dict()
  files = os.listdir(rawDataPath)
  files = [file for file in files if file.endswith(".mp3")]
  nbFiles = len(files)

  # Create path if not existing
  if not os.path.exists(os.path.dirname(spectrogramsPath)):
    try:
      os.makedirs(os.path.dirname(spectrogramsPath))
    except OSError as exc: # Guard against race condition
      # pylint: disable=undefined-variable
      if exc.errno != errno.EEXIST:
        raise

  # Rename files according to genre
  for index, filename in enumerate(files):
    # Rename file before finding genre
    # Strip out any special characters
    os.rename(rawDataPath+filename, rawDataPath+re.sub('[^A-Za-z0-9. ]+', '', filename))
    filename = re.sub('[^A-Za-z0-9. ]+', '', filename)

    print("Creating spectrogram for file {}/{}...  {}".format(index+1, nbFiles, filename))

    fileGenre = getGenre(rawDataPath + filename)
    genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
    fileID = genresID[fileGenre]
    newFilename = fileGenre+"_"+str(fileID) # if fileGenre is byte then do this
    createSpectrogram(filename, newFilename)


def createSlicesFromAudio():
  '''Whole pipeline .mp3 -> .png slices'''
  print("Creating spectrograms...")
  createSpectrogramsFromAudio()
  print("Spectrograms created!")

  print("Creating slices...")
  createSlicesFromSpectrograms(sliceXSize, sliceYSize)
  print("Slices created!")
