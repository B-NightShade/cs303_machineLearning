# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:26:26 2022

@author: weste
"""


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import normalize

data = pd.read_csv("train.tsv", delimiter="\t")

colors = {'Very_Low' : 'red', 'Low' : 'green', 'Middle':'purple', 'High':'cyan'}
y = data['UNS']
c = [colors[x] for x in y]
pd.plotting.scatter_matrix(data, c=c)
'''
X = data.drop(['UNS'], axis = 1)
allThree = KNeighborsClassifier(n_neighbors=3)
allThree.fit(X, y)
result = KNeighborsClassifier()


pred = cross_val_predict(result, X, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='micro')
r = recall_score(y, pred, average='micro')

print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))

allFive = KNeighborsClassifier(n_neighbors=2)
allFive.fit(X, y)
result2 = KNeighborsClassifier()


pred2 = cross_val_predict(result2, X, y, cv=5)
a = accuracy_score(y, pred2)
p = precision_score(y, pred2, average='micro')
r = recall_score(y, pred2, average='micro')

print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))

X2 = data.drop(['UNS'], axis = 1)
X2 = X2.drop(['STG'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X2, y)
result2 = KNeighborsClassifier()


pred2 = cross_val_predict(result2, X2, y, cv=5)
a2 = accuracy_score(y, pred2)
p2 = precision_score(y, pred2, average='micro')
r2 = recall_score(y, pred, average='micro')

print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))

X3 = data.drop(['UNS'], axis = 1)
X3 = X3.drop(['STR'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X3, y)
result3 = KNeighborsClassifier()


pred2 = cross_val_predict(result3, X3, y, cv=5)
a3 = accuracy_score(y, pred2)
p3 = precision_score(y, pred2, average='micro')
r3 = recall_score(y, pred, average='micro')

print("accuracy: "+ str(a3))
print("percision: "+ str(p3))
print("recall: "+ str(r3))

X4 = data.drop(['UNS'], axis = 1)
X4 = X4.drop(['STG'], axis = 1)
X4 = X4.drop(['STR'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X4, y)
result4 = KNeighborsClassifier()


pred4 = cross_val_predict(result4, X4, y, cv=5)
a4 = accuracy_score(y, pred4)
p4 = precision_score(y, pred4, average='micro')
r4 = recall_score(y, pred4, average='micro')

print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")
'''
X = data.drop(['UNS'], axis = 1)
allThree = KNeighborsClassifier(n_neighbors=3)
allThree.fit(X, y)
result = KNeighborsClassifier()


pred = cross_val_predict(result, X, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='micro')
r = recall_score(y, pred, average='micro')

print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

X2 = data.drop(['UNS'], axis = 1)
X2 = X2.drop(['STG'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X2, y)
result2 = KNeighborsClassifier()


pred2 = cross_val_predict(result2, X2, y, cv=5)
a2 = accuracy_score(y, pred2)
p2 = precision_score(y, pred2, average='micro')
r2 = recall_score(y, pred2, average='micro')

print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

X3 = data.drop(['UNS'], axis = 1)
X3 = X3.drop(['STR'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X3, y)
result3 = KNeighborsClassifier()


pred3 = cross_val_predict(result3, X3, y, cv=5)
a3 = accuracy_score(y, pred3)
p3 = precision_score(y, pred3, average='micro')
r3 = recall_score(y, pred3, average='micro')

print("accuracy: "+ str(a3))
print("percision: "+ str(p3))
print("recall: "+ str(r3))
print("\n")

X4 = data.drop(['UNS'], axis = 1)
X4 = X4.drop(['STG'], axis = 1)
X4 = X4.drop(['STR'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X4, y)
result4 = KNeighborsClassifier()


pred4 = cross_val_predict(result4, X4, y, cv=5)
a4 = accuracy_score(y, pred4)
p4 = precision_score(y, pred4, average='micro')
r4 = recall_score(y, pred4, average='micro')

print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")

X5 = data.drop(['UNS'], axis = 1)
X5 = X5.drop(['SCG'], axis = 1)
X5 = X5.drop(['STR'], axis = 1)
dropSTG = KNeighborsClassifier(n_neighbors=3)
dropSTG.fit(X5, y)
result5 = KNeighborsClassifier()


pred5 = cross_val_predict(result5, X5, y, cv=5)
a5 = accuracy_score(y, pred5)
p5 = precision_score(y, pred5, average='micro')
r5 = recall_score(y, pred5, average='micro')

print("accuracy: "+ str(a5))
print("percision: "+ str(p5))
print("recall: "+ str(r5))
print("\n")


X6 = data.drop(['UNS'], axis = 1)
X6 = X6.drop(['STG'], axis = 1)
X6 = X6.drop(['STR'], axis = 1)
X6 = X6.drop(['SCG'], axis = 1)
X6 = normalize(X6)
df6 = pd.DataFrame(X6, columns = ["LPR", "PEG"])

dropSTG = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
dropSTG.fit(df6, y)
result6 = KNeighborsClassifier()


pred6 = cross_val_predict(result6, df6, y, cv=5)
a6 = accuracy_score(y, pred6)
p6 = precision_score(y, pred6, average='micro')
r6 = recall_score(y, pred6, average='micro')

print("accuracy: "+ str(a6))
print("percision: "+ str(p6))
print("recall: "+ str(r6))
print("\n")



tData = pd.read_csv("test.tsv", delimiter="\t")
final = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
final.fit(X6, y)
alteredTest = tData.drop(['STG'], axis = 1)
alteredTest2 = alteredTest.drop(['STR'], axis = 1)
alteredTest3 = alteredTest2.drop(['SCG'], axis = 1)
testLabels = final.predict(alteredTest3)

tData['label'] = testLabels
colors = {'Very_Low' : 'red', 'Low' : 'green', 'Middle':'purple', 'High':'cyan'}
y = tData['label']
c2 = [colors[x] for x in y]
pd.plotting.scatter_matrix(tData, c=c2)
