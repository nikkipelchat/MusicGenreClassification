# Deep Audio Classification
A pipeline to build a dataset from your own music library and use it to fill the missing genres

Forked from a repo based on this [article on Medium](https://medium.com/@juliendespois/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194#.yhemoyql0)

[My paper published](https://ieeexplore.ieee.org/document/8861555) on this repo

Required install:

```
eyed3
sox --with-lame
tensorflow
tflearn
```

- Create folder Data/Raw/
- Place your labeled .mp3 files in Data/Raw/

To run lint on all python files:

```
pylint *.py
```

To create the song slices (might be long):

```
python main.py slice
```

To train the classifier (long too):

```
python main.py train
```

To test the classifier (fast):

```
python main.py test
```

- All editable parameters are in the config.py file
- I haven't implemented the pipeline to label new songs with the model, but that can be easily done with the provided functions, and eyed3 for the mp3 manipulation. Here's the full pipeline you would need to use.

![alt tag](https://github.com/despoisj/DeepAudioClassification/blob/master/img/pipeline.png)
