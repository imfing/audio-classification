# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Prepare the data
X = np.load('feat.npy')
y = np.load('label.npy').ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=233)

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

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Plot model
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train - 1, num_classes=10)
y_test = keras.utils.to_categorical(y_test - 1, num_classes=10)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model.fit(X_train, y_train, batch_size=64, epochs=1000)
score, acc = model.evaluate(X_test, y_test, batch_size=16)

print('Test score:', score)
print('Test accuracy:', acc)
