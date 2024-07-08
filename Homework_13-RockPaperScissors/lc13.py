# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:16:19 2022

@author: weste
"""

import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

def CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 32,
                     kernel_size = (5,5),
                     padding='valid',
                     input_shape=(28,28,1),
                     data_format='channels_last',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

(X,y), (Xt,_) = mnist.load_data()
X=X.reshape((X.shape[0],28,28,1))
Xt = Xt.reshape((Xt.shape[0],28,28,1))

X = X/255
Xt=Xt/255

y = np_utils.to_categorical(y)
num_classes = y.shape[1]

cnn = CNN_model(num_classes)
cnn.fit(X,y)

yt = cnn.predict(Xt)

i = random.randint(0,Xt.shape[0])
xt = Xt[i]
yp = yt[i]
label = np.argmax(yp)

xt = xt.reshape((28,28))
plt.imshow(xt)
plt.title(label)
plt.show()


