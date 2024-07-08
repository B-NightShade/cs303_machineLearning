# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:56:17 2022

@author: weste
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv("train.csv")

numColumns = data.select_dtypes(include='int')

for column, data2 in numColumns.items():
    if (column != 'age'):
            corellation = np.corrcoef(data2, data['age'])
            print("compare with age")
            print(column, " corellation: ")
            print(corellation)
            print("\n")
    if (column != 'age'):
            corellation = np.corrcoef(data2, data['education-num'])
            print("compare with education num")
            print(column, " corellation: ")
            print(corellation)
            print("\n")
    if (column != 'hours-per-week'):
            corellation = np.corrcoef(data2, data['hours-per-week'])
            print("hours-per-week")
            print(column, " corellation: ")
            print(corellation)
            print("\n")
    
pd.plotting.scatter_matrix(numColumns)

print("non na values\n", data.count())


stgov =  np.where((data['workclass']=='State-gov') & (data['income'] == '>50K'))

total = 0
values, counts =np.unique(stgov, return_counts=True)
for count in counts:
    total += int(count)

lfincome =  np.where((data['sex']=='Female') & (data['income'] == '<=50K'))

total2 = 0
values, counts =np.unique(lfincome, return_counts=True)
for count in counts:
    total2 += int(count)
    

hmincome =  np.where((data['sex']=='Male') & (data['income'] == '>50K'))

total3 = 0
values, counts =np.unique(hmincome, return_counts=True)
for count in counts:
    total3 += int(count)

lmincome =  np.where((data['sex']=='Male') & (data['income'] == '<=50K'))

total4 = 0
values, counts =np.unique(lmincome, return_counts=True)
for count in counts:
    total4 += int(count)

men = data.loc[data['sex'] == 'Male']
women = data.loc[data['sex']=='Female']

a = np.shape(women)[0]
b = np.shape(men)[0]

perHf = (total / a)*100
perlf = (total2 / a)*100
perHm = (total3 / b)*100
perlm = (total4 / b) *100

plt.figure()
plt.title("sex vs income")
plt.xlabel("category of income")
plt.ylabel("percent in that group")
plt.bar('highFincome',perHf , color ='black', label = "fem high income")
plt.bar('highMincome',perHm , color ='red', label = "men high income")
plt.bar('lowFincome',perlf , color ='cyan', label = "fem low income")
plt.bar('lowMincome',perlm , color ='green', label = "men low income")
plt.legend()
plt.show()

hfincome =  np.where((data['sex']=='Female') & (data['income'] == '>50K'))

total = 0
values, counts =np.unique(hfincome, return_counts=True)
for count in counts:
    total += int(count)

lfincome =  np.where((data['sex']=='Female') & (data['income'] == '<=50K'))

total2 = 0
values, counts =np.unique(lfincome, return_counts=True)
for count in counts:
    total2 += int(count)
    

hmincome =  np.where((data['sex']=='Male') & (data['income'] == '>50K'))

total3 = 0
values, counts =np.unique(hmincome, return_counts=True)
for count in counts:
    total3 += int(count)

lmincome =  np.where((data['sex']=='Male') & (data['income'] == '<=50K'))

total4 = 0
values, counts =np.unique(lmincome, return_counts=True)
for count in counts:
    total4 += int(count)

men = data.loc[data['sex'] == 'Male']
women = data.loc[data['sex']=='Female']

a = np.shape(women)[0]
b = np.shape(men)[0]

perHf = (total / a)*100
perlf = (total2 / a)*100
perHm = (total3 / b)*100
perlm = (total4 / b) *100

plt.figure()
plt.title("sex vs income")
plt.xlabel("category of income")
plt.ylabel("percent in that group")
plt.bar('highFincome',perHf , color ='black', label = "fem high income")
plt.bar('highMincome',perHm , color ='red', label = "men high income")
plt.bar('lowFincome',perlf , color ='cyan', label = "fem low income")
plt.bar('lowMincome',perlm , color ='green', label = "men low income")
plt.legend()
plt.show()



y = data['income']
X = data.drop(['income'], axis = 1)

categories = pd.Categorical(X['education']) 
X['education'] = categories.codes 
categories = pd.Categorical(X['workclass']) 
X['workclass'] = categories.codes 
categories = pd.Categorical(X['marital-status']) 
X['marital-status'] = categories.codes 
categories = pd.Categorical(X['occupation']) 
X['occupation'] = categories.codes
categories = pd.Categorical(X['relationship']) 
X['relationship'] = categories.codes
categories = pd.Categorical(X['race']) 
X['race'] = categories.codes
categories = pd.Categorical(X['sex']) 
X['sex'] = categories.codes
categories = pd.Categorical(X['native-country']) 
X['native-country'] = categories.codes


NBG = GaussianNB()
result = NBG.fit(X,y)

pred = cross_val_predict(result, X, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='macro')
r = recall_score(y, pred, average='macro')

print("all columns gaussianNB")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

NBG = BernoulliNB()
result = NBG.fit(X,y)

pred = cross_val_predict(result, X, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='macro')
r = recall_score(y, pred, average='macro')

print("all columns BernoulliNB")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

NBG = ComplementNB()
result = NBG.fit(X,y)

pred = cross_val_predict(result, X, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='macro')
r = recall_score(y, pred, average='macro')

print("all columns ComplementNB")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

NBG = GaussianNB()
X2 = X.drop(['relationship'], axis = 1)
result = NBG.fit(X2,y)

pred = cross_val_predict(result, X2, y, cv=5)
a2 = accuracy_score(y, pred)
p2 = precision_score(y, pred, average='macro')
r2 = recall_score(y, pred, average='macro')

print("drop relationship gaussianNB")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

NBG = GaussianNB()
X3 = X2.drop(['native-country'], axis = 1)
result = NBG.fit(X3,y)

pred = cross_val_predict(result, X3, y, cv=5)
a2 = accuracy_score(y, pred)
p2 = precision_score(y, pred, average='macro')
r2 = recall_score(y, pred, average='macro')

print("drop native-country and relationship gaussianNB")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")


NBG = GaussianNB()
X4 = X3.drop(['race'], axis = 1)
result = NBG.fit(X4,y)

pred = cross_val_predict(result, X4, y, cv=5)
a2 = accuracy_score(y, pred)
p2 = precision_score(y, pred, average='macro')
r2 = recall_score(y, pred, average='macro')

print("drop native-country, race, relationship gaussianNB")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

NBG = GaussianNB()
X4 = X3.drop(['marital-status'], axis = 1)
result = NBG.fit(X4,y)

pred = cross_val_predict(result, X4, y, cv=5)
a2 = accuracy_score(y, pred)
p2 = precision_score(y, pred, average='macro')
r2 = recall_score(y, pred, average='macro')

print("drop native-country, marital status, relationship gaussianNB")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

NBG = GaussianNB()
X5 = X.drop(['sex'], axis = 1)
X5 = X5.drop(['native-country'], axis = 1)
X5 = X5.drop(['relationship'], axis =1)
result = NBG.fit(X5,y)

test = pd.read_csv("test.csv")

alteredTest = test.drop(['sex'], axis = 1)
alteredTest = alteredTest.drop(['native-country'], axis = 1)
alteredTest = alteredTest.drop(['relationship'], axis =1)

categories = pd.Categorical(alteredTest['education']) 
alteredTest['education'] = categories.codes 
categories = pd.Categorical(alteredTest['workclass']) 
alteredTest['workclass'] = categories.codes 
categories = pd.Categorical(alteredTest['marital-status']) 
alteredTest['marital-status'] = categories.codes 
categories = pd.Categorical(alteredTest['occupation']) 
alteredTest['occupation'] = categories.codes
categories = pd.Categorical(alteredTest['race']) 
alteredTest['race'] = categories.codes


testLabels = result.predict(alteredTest)

test['income'] = testLabels

'''
X5 = X[['race', 'sex', 'education', 'workclass', 'occupation', 'hours-per-week']]
result = NBG.fit(X5,y)

pred = cross_val_predict(result, X5, y, cv=5)
a3 = accuracy_score(y, pred)
p3 = precision_score(y, pred, average='macro')
r3 = recall_score(y, pred, average='macro')

print("drop native-country, race, relationship gaussianNB")
print("accuracy: "+ str(a3))
print("percision: "+ str(p3))
print("recall: "+ str(r3))
print("\n")
'''