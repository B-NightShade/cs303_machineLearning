# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:47:42 2022

@author: weste
"""

import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

trainData = pd.read_csv("train.csv", delimiter = ',')

y= trainData['class']
X = trainData.drop(['class'], axis = 1)

print(np.unique(y))

np.random.seed(15)

label = X.loc[trainData['class'] == 0]

ranNum1 = np.random.randint(label.shape[0]) 

x1 = label.iloc[[ranNum1]]
x1 = pd.DataFrame.to_numpy(x1)
x1 = np.reshape(x1,(32,32))
plt.imshow(x1)
plt.title("ran row image of class 0")
plt.show()


label1 = X.loc[trainData['class'] == 1]

ranNum2 = np.random.randint(label1.shape[0]) 

x = label1.iloc[[ranNum2]]
x = pd.DataFrame.to_numpy(x)
x = np.reshape(x,(32,32))
plt.imshow(x)
plt.title("ran row image of class 1")
plt.show()

label2 = X.loc[trainData['class'] == 2]

ranNum3 = np.random.randint(label2.shape[0]) 

x2 = label2.iloc[[ranNum3]]
x2 = pd.DataFrame.to_numpy(x2)
x2 = np.reshape(x2,(32,32))
plt.imshow(x2)
plt.title("ran row image of class 2")
plt.show()
'''
def CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 32,
                     kernel_size = (5,5),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_model2(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 45,
                     kernel_size = (8,8),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (5,5)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_model3(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 64,
                     kernel_size = (3,3),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (5,5)))
    model.add(Conv2D(filters = 32,
                     kernel_size = (3,3),
                     padding='valid',
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (4,4)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_model4(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 45,
                     kernel_size = (8,8),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (5,5)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], steps_per_execution=(5))
    return model

def CNN_model5(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 45,
                     kernel_size = (8,8),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (5,5)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], steps_per_execution=(14))
    return model
'''
def CNN_modelF(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 45,
                     kernel_size = (8,8),
                     padding='valid',
                     input_shape=(32,32,1),
                     data_format='channels_last',
                     activation='tanh'))
    model.add(MaxPooling2D(pool_size = (5,5)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], steps_per_execution=(14))
    return model



X = pd.DataFrame.to_numpy(X)
print(len(X))
'''
X = X.reshape(len(X),32,32,1) 


X = X/255
y=np_utils.to_categorical(y)
num_classes = y.shape[1]
'''

'''
cnn = CNN_model(num_classes)
cnn.fit(X,y)

cnn2 = CNN_model2(num_classes)
cnn2.fit(X,y)

cnn3 = CNN_model3(num_classes)
cnn3.fit(X,y)

cnn4 = CNN_model4(num_classes)
cnn4.fit(X,y, epochs = 7)

y2= trainData['class']
y_pred2 = cnn4.predict(X)
y_pred2 = np.argmax(y_pred2,axis = 1)
print(classification_report(y2, y_pred2))

cnn5 = CNN_model4(num_classes)
cnn5.fit(X,y, epochs = 14)

y_pred = cnn5.predict(X)
y_pred = np.argmax(y_pred,axis = 1)
print(classification_report(y2, y_pred))

FinalCNN = CNN_modelF(num_classes)
FinalCNN.fit(X,y, epochs = 14) 

testData = pd.read_csv("test.csv", delimiter = ',')

altTest = pd.DataFrame.to_numpy(testData)
altTest = altTest.reshape(len(altTest),32,32,1) 


altTest = altTest/255

labels = FinalCNN.predict(altTest)
labels = np.argmax(labels,axis = 1)

testData['class'] = labels


yt= testData['class']
Xt = testData.drop(['class'], axis = 1)

print(np.unique(yt))

np.random.seed(22)

label = Xt.loc[testData['class'] == 0]

ranNum1 = np.random.randint(label.shape[0]) 

x1 = label.iloc[[ranNum1]]
x1 = pd.DataFrame.to_numpy(x1)
x1 = np.reshape(x1,(32,32))
plt.imshow(x1)
plt.title("ran row image of class 0 test data")
plt.show()


label1 = Xt.loc[testData['class'] == 1]

ranNum2 = np.random.randint(label1.shape[0]) 

x = label1.iloc[[ranNum2]]
x = pd.DataFrame.to_numpy(x)
x = np.reshape(x,(32,32))
plt.imshow(x)
plt.title("ran row image of class 1 test data")
plt.show()

label2 = Xt.loc[testData['class'] == 2]

ranNum3 = np.random.randint(label2.shape[0]) 

x2 = label2.iloc[[ranNum3]]
x2 = pd.DataFrame.to_numpy(x2)
x2 = np.reshape(x2,(32,32))
plt.imshow(x2)
plt.title("ran row image of class 2 test data")
plt.show()

label = Xt.loc[testData['class'] == 0]

ranNum1 = np.random.randint(label.shape[0]) 

x1 = label.iloc[[ranNum1]]
x1 = pd.DataFrame.to_numpy(x1)
x1 = np.reshape(x1,(32,32))
plt.imshow(x1)
plt.title("ran row image of class 0 test data")
plt.show()


label1 = Xt.loc[testData['class'] == 1]

ranNum2 = np.random.randint(label1.shape[0]) 

x = label1.iloc[[ranNum2]]
x = pd.DataFrame.to_numpy(x)
x = np.reshape(x,(32,32))
plt.imshow(x)
plt.title("ran row image of class 1 test data")
plt.show()

label2 = Xt.loc[testData['class'] == 2]

ranNum3 = np.random.randint(label2.shape[0]) 

x2 = label2.iloc[[ranNum3]]
x2 = pd.DataFrame.to_numpy(x2)
x2 = np.reshape(x2,(32,32))
plt.imshow(x2)
plt.title("ran row image of class 2 test data")
plt.show()
'''