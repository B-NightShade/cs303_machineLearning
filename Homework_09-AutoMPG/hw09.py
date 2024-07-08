# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:58:31 2022

@author: weste
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


trainData = pd.read_csv("train.csv", delimiter = ",")

print("non na values\n", trainData.count())

cylinder = trainData['cylinders']
dis = trainData['displacement']
horse = trainData['horsepower']
w = trainData['weight']
a = trainData['acceleration']
m = trainData['model']
o = trainData['origin']
mpg = trainData['mpg']

plt.scatter(mpg, cylinder)
plt.xlabel("mpg")
plt.ylabel("cyliders")
plt.title("mpg vs cylinders")
plt.show()

plt.scatter(mpg, dis)
plt.xlabel("mpg")
plt.ylabel("displacement")
plt.title("mpg vs displacement")
plt.show()

plt.scatter(mpg, horse)
plt.xlabel("mpg")
plt.ylabel("horsepower")
plt.title("mpg vs horsepower")
plt.show()

plt.scatter(mpg, w)
plt.xlabel("mpg")
plt.ylabel("weight")
plt.title("mpg vs weight")
plt.show()

plt.scatter(mpg, a)
plt.xlabel("mpg")
plt.ylabel("acceleration")
plt.title("mpg vs acceleration")
plt.show()

plt.scatter(mpg, m)
plt.xlabel("mpg")
plt.ylabel("model")
plt.title("mpg vs model")
plt.show()

plt.scatter(mpg, o)
plt.xlabel("mpg")
plt.ylabel("origin")
plt.title("mpg vs orgin")
plt.show()


print("average cylinder: " + str(np.mean(cylinder)))
print("average displacement: " + str(np.mean(dis)))
print("average horsepower: " + str(np.mean(horse)))
print("average weight: " + str(np.mean(w)))
print("average acceleration: " + str(np.mean(a)))
print("average model: " + str(np.mean(m)))
print("average origin: " + str(np.mean(o)))

trainData = trainData.dropna()
X = trainData.drop(['mpg'], axis = 1)
X = X.drop(['car'], axis =1)

X=X['horsepower']
y=trainData['mpg']
m, b, r, _, _ = stats.linregress(X,y)
print("R^2 =", r**2)


yp = m * X + b
plt.plot(X,y, 'go')
plt.plot(X,yp, color='red')
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.title("linear regression horsepower")
plt.show()

X6 = trainData.drop(['mpg'], axis = 1)
X6 = X6.drop(['car'], axis =1)


X6=X6['acceleration']
y=trainData['mpg']
m, b, r, _, _ = stats.linregress(X6,y)
print("R^2 =", r**2)


yp = m * X6 + b
plt.plot(X6,y, 'go')
plt.plot(X6,yp, color='red')
plt.xlabel("acceleration")
plt.ylabel("mpg")
plt.title("linear regression acceleration")
plt.show()

X6 = trainData.drop(['mpg'], axis = 1)
X6 = X6.drop(['car'], axis =1)

X6=X6['weight']
y=trainData['mpg']
m, b, r, _, _ = stats.linregress(X6,y)
print("R^2 =", r**2)


yp = m * X6 + b
plt.plot(X6,y, 'go')
plt.plot(X6,yp, color='red')
plt.xlabel("weight")
plt.ylabel("mpg")
plt.title("linear regression weight")
plt.show()




X2 = trainData.drop(['mpg'], axis = 1)
X2 = X2.drop(['car'], axis =1)

regress = LinearRegression()
regress.fit(X2,y)
pred = cross_val_predict(regress, X2, y, cv=5)
r2 = r2_score(y, pred)
print("R^2: ", r2)
mse = mean_squared_error(y, pred)
print("mean squared error: ", mse)

X3 = trainData.drop(['mpg'], axis = 1)
X3 = X3.drop(['car'], axis =1)
X3 = X3.drop(['origin'], axis = 1)

regress = LinearRegression()
regress.fit(X3,y)

pred3 = cross_val_predict(regress, X3, y, cv=5)
r2 = r2_score(y, pred3)
print("R^2: ", r2)
mse = mean_squared_error(y, pred3)
print("mean squared error: ", mse)

X4 = trainData.drop(['mpg'], axis = 1)
X4 = X4.drop(['car'], axis =1)
X4 = X4.drop(['cylinders'], axis = 1)
X4 = X4.drop(['origin'], axis = 1)

regress = LinearRegression()
regress.fit(X4,y)

pred2 = cross_val_predict(regress, X4, y, cv=5)
r2 = r2_score(y, pred2)
print("R^2: ", r2)
mse = mean_squared_error(y, pred2)
print("mean squared error: ", mse)


X5 = trainData.drop(['mpg'], axis = 1)
X5 = X5.drop(['car'], axis =1)

regress2 = LinearRegression(normalize=True)
regress2.fit(X5,y)
pred4 = cross_val_predict(regress2, X5, y, cv=5)
r2 = r2_score(y, pred4)
print("R^2: ", r2)
mse = mean_squared_error(y, pred4)
print("mean squared error: ", mse)



X7 = trainData.drop(['mpg'], axis = 1)
X7 = X7.drop(['car'], axis =1)

X7 = X7[['weight','horsepower', 'acceleration', 'displacement']]
regress2 = LinearRegression(normalize=True)
regress2.fit(X7,y)
pred5 = cross_val_predict(regress2, X7, y, cv=5)
r2 = r2_score(y, pred5)
print("R^2: ", r2)
mse = mean_squared_error(y, pred5)
print("mean squared error: ", mse)



X2 = trainData.drop(['mpg'], axis = 1)
X2 = X2.drop(['car'], axis =1)

regress = LinearRegression()
regress.fit(X2,y)
predMPG = cross_val_predict(regress, X2, y, cv=5)
print("mean pred", np.mean(predMPG))
print("mean original", np.mean(trainData['mpg']))
print("std pred", np.std(predMPG))
print("std original", np.std(trainData['mpg']))
print("max pred", np.max(predMPG))
print("max original", np.max(trainData['mpg']))
print("min pred", np.min(predMPG))
print("min original", np.min(trainData['mpg']))


Xfinal = trainData.drop(['mpg'], axis = 1)
Xfinal = Xfinal.drop(['car'], axis =1)

Fregress = LinearRegression()
Fregress.fit(Xfinal,y)

testData = pd.read_csv("test.csv")
testData = testData.dropna()
altTest = testData.drop(['car'], axis = 1)

mpg = Fregress.predict(altTest)

testData['mpg'] = mpg