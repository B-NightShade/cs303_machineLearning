# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:06:01 2022

@author: weste
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score

names = ["slength", "swidth", "plength", "pwidth", "class"]
data = pd.read_csv("iris.data", delimiter = "," , names = names)

pd.plotting.scatter_matrix(data)

X = data.drop(['class'], axis = 1)
y = data['class']

myDtree = tree.DecisionTreeClassifier()

AccuracyVal = cross_val_score(myDtree, X, y, cv=5)

pred = cross_val_predict(myDtree, X, y, cv=5)

a = accuracy_score(y, pred)
p = precision_score(y, pred, average='micro')
r = recall_score(y, pred, average='mirco')

print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))