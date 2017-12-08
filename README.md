# Environmental Sound Classification using Deep Learning

> A project from Digital Signal Processing course

## Dependencies

- Python 2.7 (3.6 not tested)
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

## Feature Extraction

Put audio files under `data` directory and run the following command:

```shell
$ python feat_extract.py
```

Features and labels will be generated and saved in the directory.

## Classify with SVM

Make sure you have `scikit-learn` installed and `feat.npy` and `label.npy` under the same directory. Run `svm.py` and you could see the result.

## Classify with Multilayer Perception

Install `tensorflow` and `keras` at first. Run `nn.py` to train and test the network.

## Classify with Convolutional Neural Network

Run `cnn.py` to train and test a CNN.