#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import time
import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-t', '--train',             action='store_true', help='train neural network with extracted features')
parser.add_argument('-e', '--epochs', type=int, metavar='N',default=1000,   help='epochs to train')
parser.add_argument('-p', '--predict',           action='store_true', help='predict files in ./predict folder')
parser.add_argument('-P', '--real-time-predict', action='store_true', help='predict sound in real time')
args = parser.parse_args()

if args.train: 
    # Prepare the data
    args = parser.parse_args()
    if not os.path.exists('feat.npy') or not os.path.exists('label.npy'): 
        print('Run feat_extract.py first')
        sys.exit(1)
    else:
        X = np.load('feat.npy')
        y = np.load('label.npy').ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=233)

    # Build the Neural Network
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # Convert label to onehot
    y_train = keras.utils.to_categorical(y_train - 1, num_classes=10)
    y_test = keras.utils.to_categorical(y_test - 1, num_classes=10)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    start = time.time()
    model.fit(X_train, y_train, batch_size=64, epochs=args.epochs)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Training took: %d seconds' % int(time.time() - start))
    model.save('trained_model.h5')

elif args.predict: 
    model = keras.models.load_model('trained_model.h5')
    X_predict = np.load('predict_feat.npy')
    filenames = np.load('predict_filenames.npy')
    X_predict = np.expand_dims(X_predict, axis=2)
    pred = model.predict_classes(X_predict)
    for pair in list(zip(filenames, pred)): print(pair)

elif args.real_time_predict: 
    import sounddevice as sd
    import soundfile as sf
    import queue
    import librosa
    from feat_extract import *
    import sys
    model = keras.models.load_model('trained_model.h5')
    while True:
        try:
            features = np.empty((0,193))
            mfccs, chroma, mel, contrast,tonnetz = extract_feature()
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            features = np.expand_dims(features, axis=2)
            pred = model.predict_classes(features)
            for p in pred: 
                print(p)
                sys.stdout.flush()
        except KeyboardInterrupt: parser.exit(0)
        except Exception as e: parser.exit(type(e).__name__ + ': ' + str(e))
