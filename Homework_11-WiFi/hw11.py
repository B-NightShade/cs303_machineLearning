# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:35:34 2022

@author: weste
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score

trainData = pd.read_csv("train.csv", delimiter=',')

atrA = trainData['A']
atrB = trainData['B']
atrC = trainData['C']
atrD = trainData['D']
atrE = trainData['E']
atrF = trainData['F']
atrG = trainData['G']

y = trainData['Location']
X = trainData.drop(['Location'], axis = 1)
'''
fig =seaborn.scatterplot(atrA, atrB, hue=y, palette=("deep"))
fig.set(title = "A vs B color locations")
plt.show()

fig =seaborn.scatterplot(atrA, atrC, hue=y, palette=("deep"))
fig.set(title = "A vs C color locations")
plt.show()

fig =seaborn.scatterplot(atrA, atrD, hue=y, palette=("deep"))
fig.set(title = "A vs D color locations")
plt.show()

fig =seaborn.scatterplot(atrA, atrE, hue=y, palette=("deep"))
fig.set(title = "A vs E color locations")
plt.show()

fig =seaborn.scatterplot(atrA, atrF, hue=y, palette=("deep"))
fig.set(title = "A vs F color locations")
plt.show()

fig =seaborn.scatterplot(atrA, atrG, hue=y, palette=("deep"))
fig.set(title = "A vs G color locations")
plt.show()

fig =seaborn.scatterplot(atrB, atrC, hue=y, palette=("deep"))
fig.set(title = "B vs C color locations")
plt.show()

fig =seaborn.scatterplot(atrB, atrD, hue=y, palette=("deep"))
fig.set(title = "B vs D color locations")
plt.show()

fig =seaborn.scatterplot(atrB, atrE, hue=y, palette=("deep"))
fig.set(title = "B vs E color locations")
plt.show()

fig =seaborn.scatterplot(atrB, atrF, hue=y, palette=("deep"))
fig.set(title = "B vs F color locations")
plt.show()

fig =seaborn.scatterplot(atrB, atrG, hue=y, palette=("deep"))
fig.set(title = "B vs G color locations")
plt.show()

fig =seaborn.scatterplot(atrC, atrD, hue=y, palette=("deep"))
fig.set(title = "C vs D color locations")
plt.show()

fig =seaborn.scatterplot(atrC, atrE, hue=y, palette=("deep"))
fig.set(title = "C vs E color locations")
plt.show()

fig =seaborn.scatterplot(atrC, atrF, hue=y, palette=("deep"))
fig.set(title = "C vs F color locations")
plt.show()

fig =seaborn.scatterplot(atrC, atrG, hue=y, palette=("deep"))
fig.set(title = "C vs G color locations")
plt.show()

fig =seaborn.scatterplot(atrD, atrE, hue=y, palette=("deep"))
fig.set(title = "D vs E color locations")
plt.show()

fig =seaborn.scatterplot(atrD, atrF, hue=y, palette=("deep"))
fig.set(title = "D vs F color locations")
plt.show()

fig =seaborn.scatterplot(atrD, atrG, hue=y, palette=("deep"))
fig.set(title = "D vs G color locations")
plt.show()

fig =seaborn.scatterplot(atrE, atrF, hue=y, palette=("deep"))
fig.set(title = "E vs F color locations")
plt.show()

fig =seaborn.scatterplot(atrE, atrG, hue=y, palette=("deep"))
fig.set(title = "E vs G color locations")
plt.show()

fig =seaborn.scatterplot(atrF, atrG, hue=y, palette=("deep"))
fig.set(title = "F vs G color locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrA)
fig.set(title = "A boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrB)
fig.set(title = "B boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrC)
fig.set(title = "C boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrD)
fig.set(title = "D boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrE)
fig.set(title = "E boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrF)
fig.set(title = "F boxplots of locations")
plt.show()

fig = seaborn.boxplot(x = y, y = atrG)
fig.set(title = "G boxplots of locations")
plt.show()
'''

model = SVC()
model.fit(X, y)

y_pred = cross_val_predict(model, X, y, cv=5)
a = accuracy_score(y, y_pred)
p = precision_score(y, y_pred, average='macro')
r = recall_score(y, y_pred, average='macro')

print("all attributes-generic")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

model2 = SVC(kernel='poly')
model2.fit(X, y)

y_pred2 = cross_val_predict(model2, X, y, cv=5)
a2 = accuracy_score(y, y_pred2)
p2 = precision_score(y, y_pred2, average='macro')
r2 = recall_score(y, y_pred2, average='macro')

print("all attributes- poly kernel")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

model3 = SVC(kernel='linear')
model3.fit(X, y)

y_pred3 = cross_val_predict(model3, X, y, cv=5)
a3 = accuracy_score(y, y_pred3)
p3 = precision_score(y, y_pred3, average='macro')
r3 = recall_score(y, y_pred3, average='macro')

print("all attributes- linear kernel")
print("accuracy: "+ str(a3))
print("percision: "+ str(p3))
print("recall: "+ str(r3))
print("\n")

X2 = X[['A','E','D']]
model4 = SVC()
model4.fit(X2, y)

y_pred4 = cross_val_predict(model4, X2, y, cv=5)
a4 = accuracy_score(y, y_pred4)
p4 = precision_score(y, y_pred4, average='macro')
r4 = recall_score(y, y_pred4, average='macro')

print("A E D-generic")
print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")

model2 = SVC(kernel='poly', degree = 4)
model2.fit(X, y)

y_pred2 = cross_val_predict(model2, X, y, cv=5)
a2 = accuracy_score(y, y_pred2)
p2 = precision_score(y, y_pred2, average='macro')
r2 = recall_score(y, y_pred2, average='macro')

print("all attributes- poly kernel degree 4")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")


finalModel = SVC()
finalModel.fit(X, y)

testData = pd.read_csv("test.csv", delimiter=',')

testloc = finalModel.predict(testData)

testData['Location']= testloc