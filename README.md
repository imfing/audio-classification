# Environmental Sound Classification using Deep Learning

> A project from Digital Signal Processing course
## Dependencies

- Python 3.6
- numpy
- librosa
- pysoundfile
- matplotlib
- scikit-learn
- tensorflow
- keras

## Dataset

Dataset could be downloaded at [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT) or [Github](https://github.com/karoldvl/ESC-50).

I'd recommend use ESC-10 for the sake of convenience.

Example:

```
├── 001 - Cat
│  ├── cat_1.ogg
│  ├── cat_2.ogg
│  ├── cat_3.ogg
│  ...
...
└── 002 - Dog
   ├── dog_barking_0.ogg
   ├── dog_barking_1.ogg
   ├── dog_barking_2.ogg
   ...
```

## Feature Extraction

Put audio files (`.wav` untested) under `data` directory and run the following command:

`python feat_extract.py`

Features and labels will be generated and saved in the directory.

## Classify with SVM

Make sure you have `scikit-learn` installed and `feat.npy` and `label.npy` under the same directory. Run `svm.py` and you could see the result.

## Classify with Multilayer Perceptron

Install `tensorflow` and `keras` at first. Run `nn.py` to train and test the network.

## Classify with Convolutional Neural Network

- Run `cnn.py -t` to train and test a CNN. Optionally set how many epochs to train on.
- Predict files by either:
  - Putting target files under `predict/` directory and running `cnn.py -p`
  - Recording on the fly with `cnn.py -P`
