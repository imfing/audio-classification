#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import feat_extract
from feat_extract import *
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
import os.path as op
from sklearn.model_selection import train_test_split

def train(args):
    if not op.exists('feat.npy') or not op.exists('label.npy'):
        if input('No feature/labels found. Run feat_extract.py first? (Y/n)').lower() in ['y', 'yes', '']:
            feat_extract.main()
            train(args)
    else:
        X = np.load('feat.npy')
        y = np.load('label.npy').ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=233)

    # Count the number of sub-directories in 'data'
    class_count = len(next(os.walk('data/'))[1])

    # Build the Neural Network
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Convert label to onehot
    y_train = keras.utils.to_categorical(y_train, num_classes=class_count)
    y_test = keras.utils.to_categorical(y_test, num_classes=class_count)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    start = time.time()
    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Training took: %d seconds' % int(time.time() - start))
    model.save(args.model)

def predict(args):
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        X_predict = np.expand_dims(X_predict, axis=2)
        pred = model.predict_classes(X_predict)
        for pair in list(zip(filenames, pred)): print(pair)
    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:
        train()
        predict(args)

def real_time_predict(args):
    import sounddevice as sd
    import soundfile as sf
    import queue
    import librosa
    import sys
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        while True:
            try:
                features = np.empty((0,193))
                start = time.time()
                mfccs, chroma, mel, contrast,tonnetz = extract_feature()
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                features = np.expand_dims(features, axis=2)
                pred = model.predict_classes(features)
                for p in pred:
                    print(p)
                    if args.verbose: print('Time elapsed in real time feature extraction: ', time.time() - start)
                    sys.stdout.flush()
            except KeyboardInterrupt: parser.exit(0)
            except Exception as e: parser.exit(type(e).__name__ + ': ' + str(e))
    elif input('Model not found. Train network first? (y/N)') in ['y', 'yes']:
        train()
        real_time_predict(args)


def main(args):
    if args.train: train(args)
    elif args.predict: predict(args)
    elif args.real_time_predict: real_time_predict(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network with extracted features')
    parser.add_argument('-m', '--model',             metavar='path',     default='trained_model.h5',help='use this model path on train and predict operations')
    parser.add_argument('-e', '--epochs',            metavar='N',        default=500,              help='epochs to train', type=int)
    parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files in ./predict folder')
    parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='predict sound in real time')
    parser.add_argument('-v', '--verbose',           action='store_true',                           help='verbose print')
    parser.add_argument('-s', '--log-speed',         action='store_true',                           help='performance profiling')
    parser.add_argument('-b', '--batch-size',        metavar='size',     default=64,                help='batch size', type=int)
    args = parser.parse_args()
    main(args)
