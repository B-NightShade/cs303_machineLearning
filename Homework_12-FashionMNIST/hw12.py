# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:31:06 2022

@author: weste
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

trainData = pd.read_csv("train.csv", delimiter=",")

y = trainData['label']
X = trainData.drop(['label'], axis = 1)


print(np.unique(y))
np.random.seed(12)

label = X.loc[trainData['label'] == 0]

ranNum1 = np.random.randint(label.shape[0]) 

x1 = label.iloc[[ranNum1]]
x1 = pd.DataFrame.to_numpy(x1)
x1 = np.reshape(x1,(28,28))
plt.imshow(x1)
plt.title("ran row of label 0")
plt.show()


label1 = X.loc[trainData['label'] == 1]

ranNum = np.random.randint(label1.shape[0]) 

##x = label1.iloc[[ranNum]]
x = label1.iloc[[5787]]
x = pd.DataFrame.to_numpy(x)
x = np.reshape(x,(28,28))
plt.imshow(x)
plt.title("ran row of label 1")
plt.show()

label2 = X.loc[trainData['label'] == 2]

ranNum2 = np.random.randint(label2.shape[0]) 

x2 = label2.iloc[[ranNum2]]
x2 = pd.DataFrame.to_numpy(x2)
x2 = np.reshape(x2,(28,28))
plt.imshow(x2)
plt.title("ran row of label 2")
plt.show()

label3 = X.loc[trainData['label'] == 3]

ranNum3 = np.random.randint(label3.shape[0]) 

x3 = label3.iloc[[ranNum3]]
x3 = pd.DataFrame.to_numpy(x3)
x3 = np.reshape(x3,(28,28))
plt.imshow(x3)
plt.title("ran row of label 3")
plt.show()

label4 = X.loc[trainData['label'] == 4]

ranNum4 = np.random.randint(label4.shape[0]) 

x4 = label4.iloc[[ranNum4]]
x4 = pd.DataFrame.to_numpy(x4)
x4 = np.reshape(x4,(28,28))
plt.imshow(x4)
plt.title("ran row of label 4")
plt.show()

label5 = X.loc[trainData['label'] == 5]

ranNum5 = np.random.randint(label5.shape[0]) 

x5 = label5.iloc[[ranNum5]]
x5 = pd.DataFrame.to_numpy(x5)
x5 = np.reshape(x5,(28,28))
plt.imshow(x5)
plt.title("ran row of label 5")
plt.show()

label6 = X.loc[trainData['label'] == 6]

ranNum6 = np.random.randint(label6.shape[0]) 

x6 = label6.iloc[[ranNum6]]
x6 = pd.DataFrame.to_numpy(x6)
x6 = np.reshape(x6,(28,28))
plt.imshow(x6)
plt.title("ran row of label 6")
plt.show()

label7 = X.loc[trainData['label'] == 7]

ranNum7 = np.random.randint(label7.shape[0]) 

x7 = label7.iloc[[ranNum7]]
x7 = pd.DataFrame.to_numpy(x7)
x7 = np.reshape(x7,(28,28))
plt.imshow(x7)
plt.title("ran row of label 7")
plt.show()

label8 = X.loc[trainData['label'] == 8]

ranNum8 = np.random.randint(label8.shape[0]) 

x8 = label8.iloc[[ranNum8]]
x8 = pd.DataFrame.to_numpy(x8)
x8 = np.reshape(x8,(28,28))
plt.imshow(x8)
plt.title("ran row of label 8")
plt.show()

label9 = X.loc[trainData['label'] == 9]

ranNum9 = np.random.randint(label9.shape[0]) 

x9 = label9.iloc[[ranNum9]]
x9 = pd.DataFrame.to_numpy(x9)
x9 = np.reshape(x9,(28,28))
plt.imshow(x9)
plt.title("ran row of label 9")
plt.show()

'''
def MLP_model(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def MLP_model2(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def MLP_model3(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def MLP_model4(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], steps_per_execution=(3))
    return model

def MLP_model5(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], steps_per_execution=(5))
    return model
    
X2 = pd.DataFrame.to_numpy(X)
num_pixels = 784
X2 = X2.reshape(X2.shape[0], num_pixels)
X2 = X2/255
y = np_utils.to_categorical(y)
num_classes = y.shape[1]
mlp = MLP_model(num_pixels, num_classes)
mlp.fit(X2,y)

mlp2 = MLP_model2(num_pixels, num_classes)
mlp2.fit(X2,y)

mlp3 = MLP_model3(num_pixels, num_classes)
mlp3.fit(X2,y)

mlp4 = MLP_model4(num_pixels, num_classes)
mlp4.fit(X2,y, epochs = 3)

mlp5 = MLP_model5(num_pixels, num_classes)
mlp5.fit(X2,y, epochs = 5)

def MLP_modelFinal(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], steps_per_execution=(5))
    return model


X2 = pd.DataFrame.to_numpy(X)
num_pixels = 784
X2 = X2.reshape(X2.shape[0], num_pixels)
X2 = X2/255
y = np_utils.to_categorical(y)
num_classes = y.shape[1]


mlpFinal = MLP_modelFinal(num_pixels, num_classes)
mlpFinal.fit(X2,y, epochs = 5)

testData = pd.read_csv("test.csv", delimiter = ',')
altTest = pd.DataFrame.to_numpy(testData)

altTest = altTest.reshape(altTest.shape[0], num_pixels)
altTest = altTest/255
label = mlpFinal.predict(altTest)

labels = np.argmax(label, axis = 1)
testData['label'] = labels
print(np.unique(labels))
'''