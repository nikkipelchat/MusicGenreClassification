# -*- coding: utf-8 -*-
import eyed3
import string

#Remove logs
eyed3.log.setLevel("ERROR")

def isMono(filename):
	audiofile = eyed3.load(filename)
	return audiofile.info.mode == 'Mono'

def getCategorizedGenre(genre):
	if genre == 'Hip Hop' or genre == 'HipHop':
		return 'HipHop'
	if genre == 'Rap' or genre == 'Hip HopRap' or genre == 'HipHopRap':
		return 'Rap'
	if genre == 'Electronica  Dance' or genre == 'ElectronicaDance' or genre == 'Electronic':
		return 'Electronic'
	if genre == 'Dance':
		return 'Dance'
	if genre == 'Classic Rock' or genre == 'Rock' or genre == 'General Rock':
		return 'Rock'
	if genre == 'PopClub' or genre == 'Pop Latino' or genre == 'Pop':
		return 'Pop'
	if genre == 'RBSoul' or genre == 'RB' or genre == 'Jazz' or genre == 'Reggae':
		return 'RBSoul'
	if genre == 'Alternative' or genre == 'Indie':
		return 'Alternative'
	if genre == 'Country':
		return 'Country'
	if genre == 'Classical' or genre == 'Classical Crossover':
		return 'Classical'
	# Being ignored
	if genre == 'SingerSongwriter' or genre == 'Soundtrack' or genre == 'Vocal':
		return 'Soundtrack'
	if genre == 'Other':
		return 'Other'


def getGenre(filename):
	audiofile = eyed3.load(filename)
	#No genre
	if not audiofile.tag.genre:
		return 'Other'
	else:
		translation = str.maketrans('', '', string.punctuation)
		fileGenre = audiofile.tag.genre.name.translate(translation)
		return getCategorizedGenre(fileGenre)


