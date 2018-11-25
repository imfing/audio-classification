# Whistle detection

> A project inspired by a subset of [mtobeiyf/audio-classification](https://github.com/mtobeiyf/audio-classification)

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
