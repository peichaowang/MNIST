#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.applications import VGG16
from keras.datasets import mnist
import os, sys
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import to_categorical
from keras import layers, models
from keras import optimizers
import matplotlib.pyplot as plt


def network():
    network = models.Sequential()
    network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)))
    network.add(layers.MaxPool2D(2,2))
    network.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    network.add(layers.MaxPool2D(2,2))
    network.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    
    network.add(layers.Flatten())
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(64, activation = 'relu'))
    network.add(layers.Dense(10, activation = 'softmax'))
    return network

# Prepare for training data and test data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Data normalization
train_data = train_data.reshape((60000, 28,28,1))
train_data = train_data.astype('float32')/255
test_data = test_data.reshape((10000,28,28,1))
test_data = test_data.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Get nueral network model
model = network()
model.summary()
model.compile(optimizer = optimizers.RMSprop(lr=2e-5),loss ='binary_crossentropy', metrics=['acc'])

# Show diagram
history = model.fit(train_data, train_labels, epochs=5, batch_size=64)

acc = history.history['acc']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'b', label = 'Validation accuracy')
plt.title('Validation loss and accuracy')
plt.figure()
plt.legend()
plt.show()