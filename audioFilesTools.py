'''Get genre for song after grouping them together into similar genres'''
import string
import eyed3

#Remove logs
eyed3.log.setLevel("ERROR")

def isMono(filename):
  '''Check if file is a mono channel or stereo'''
  audiofile = eyed3.load(filename)
  return audiofile.info.mode == 'Mono'


# pylint: disable=too-many-branches
def getCategorizedGenre(genre):
  '''Group similar genres'''
  result = 'Other'
  if genre in ('Hip Hop', 'HipHop'):
    result = 'HipHop'
  elif genre in ('Rap', 'RAP', 'Hip HopRap', 'HipHopRap', 'RapHipHop'):
    result = 'Rap'
  elif genre in ('Electronica  Dance', 'ElectronicaDance', 'Electronic', 'Electronica'):
    result = 'Electronic'
  elif genre in 'Dance':
    result = 'Dance' # in my music
  elif genre in 'Blues':
    result = 'Blues' # in codys music
  elif genre in ('Classic Rock', 'Rock', 'General Rock'):
    result = 'Rock'
  elif genre in ('PopClub', 'Pop Latino', 'Pop'):
    result = 'Pop'
  elif genre in ('RBSoul', 'RB', 'R&B', 'Jazz', 'Reggae'):
    result = 'RBSoul'
  elif genre in ('Alternative', 'Indie'):
    result = 'Alternative'
  elif genre in 'Country':
    result = 'Country'
  elif genre in ('Classical', 'Classical Crossover'):
    result = 'Classical'
  # Being ignored
  elif genre in ('SingerSongwriter', 'Soundtrack', 'Vocal'):
    result = 'Soundtrack'
  elif genre in 'Other':
    result = 'Other'
  return result


def getGenre(filename):
  '''Get genre of file'''
  audiofile = eyed3.load(filename)
  # No genre
  if not audiofile.tag.genre:
    print("The audio file genre is 'other'", filename)
    return 'Other'

  # audioFile has a genre
  translation = str.maketrans('', '', string.punctuation)
  fileGenre = audiofile.tag.genre.name.translate(translation)
  return getCategorizedGenre(fileGenre.strip())
