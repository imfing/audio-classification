# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Prepare the data
X =  np.load('feat.npy')
y =  np.load('label.npy').ravel()

num_classes = np.max(y, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Build the Neural Network
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=193))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train-1, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test-1, num_classes=num_classes)

# Train and test
model.fit(X_train, y_train, epochs=1000, batch_size=64)
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
