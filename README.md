# Neural Network Music Genre Classification
A CNN training on the genre of music

Code has been forked and is based on this [article on Medium](https://medium.com/@juliendespois/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194#.yhemoyql0)

[My paper](https://ieeexplore.ieee.org/document/8861555) by Nikki Pelchat which has been published about the code in this repo

Required install:
```
eyed3
sox --with-lame
tensorflow
tflearn
```

- Create folder Data/Raw/
- Place your labeled .mp3 files in Data/Raw/
- All editable parameters are in the config.py file

Avaiable Commands:

```
# To run lint on all python files
pylint *.py

# To create the song slices
python main.py slice

# To train the classifier
python main.py train

# To resume train the classifier
# This will pickup training from a checkpoint file in the folder /Checkpoint and train for 15 epochs
python main.py train --resume <checkpointNumber> --epochs 15

# To test the classifier per 2.5 second slice
python main.py test

# To test the classifier by every 2.5 second slice and then take the average vote across all of one song's slices
python main.py vote
```
