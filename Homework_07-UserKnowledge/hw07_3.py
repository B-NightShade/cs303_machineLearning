# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:12:52 2022

@author: weste
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

data = pd.read_csv("train.tsv", delimiter="\t")

colors = {'Very_Low' : 'red', 'Low' : 'green', 'Middle':'purple', 'High':'cyan'}
y = data['UNS']
c = [colors[x] for x in y]
pd.plotting.scatter_matrix(data, c=c)
plt.suptitle("red:verylow, low:green, middle:purple, high:cyan")
X = data.drop(['UNS'], axis = 1)
X = scale(X)
df = pd.DataFrame(X, columns = ["STG", "SCG", "STR", "LPR", "PEG"])


result = KNeighborsClassifier(n_neighbors=3, weights = 'distance')


pred = cross_val_predict(result, df, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='macro')
r = recall_score(y, pred, average='macro')

print("all columns and used scale to normalize data 3 neighbors")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

X2 = data.drop(['UNS'], axis = 1)
X2 = X2.drop(['STG'], axis = 1)
X2 = scale(X2)
df2 = pd.DataFrame(X2, columns = ["SCG", "STR", "LPR", "PEG"])


result2 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')


pred2 = cross_val_predict(result2, df2, y, cv=5)
a2 = accuracy_score(y, pred2)
p2 = precision_score(y, pred2, average='macro')
r2 = recall_score(y, pred2, average='macro')

print("drop stg and used scale to normalize 3 neighbors")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

X3 = data.drop(['UNS'], axis = 1)
X3 = X3.drop(['STR'], axis = 1)
X3 = scale(X3)
df3 = pd.DataFrame(X3, columns = ["STG","SCG","LPR", "PEG"])

result3 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')


pred3 = cross_val_predict(result3, df3, y, cv=5)
a3 = accuracy_score(y, pred3)
p3 = precision_score(y, pred3, average='macro')
r3 = recall_score(y, pred3, average='macro')

print("drop str use scale to normalize 3 neighbors")
print("accuracy: "+ str(a3))
print("percision: "+ str(p3))
print("recall: "+ str(r3))
print("\n")

X4 = data.drop(['UNS'], axis = 1)
X4 = X4.drop(['STG'], axis = 1)
X4 = X4.drop(['STR'], axis = 1)
X4 = scale(X4)
df4 = pd.DataFrame(X4, columns = ["SCG", "LPR", "PEG"])

result4 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')


pred4 = cross_val_predict(result4, df4, y, cv=5)
a4 = accuracy_score(y, pred4)
p4 = precision_score(y, pred4, average='macro')
r4 = recall_score(y, pred4, average='macro')

print("drop str and stg use scale to normalize 3 neighbors")
print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")

X5 = data.drop(['UNS'], axis = 1)
X5 = X5.drop(['STG'], axis = 1)
X5 = X5.drop(['STR'], axis = 1)
X5 = X5.drop(['SCG'], axis = 1)
X4 = scale(X5)
df5 = pd.DataFrame(X5, columns = ["LPR", "PEG"])


result5 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')


pred5 = cross_val_predict(result5, df5, y, cv=5)
a5 = accuracy_score(y, pred5)
p5 = precision_score(y, pred5, average='macro')
r5 = recall_score(y, pred5, average='macro')

print("drop str,scg and stg use scale to normalize 3 neightbors")
print("accuracy: "+ str(a5))
print("percision: "+ str(p5))
print("recall: "+ str(r5))
print("\n")

X6 = data.drop(['UNS'], axis = 1)
X6 = X6.drop(['STG'], axis = 1)
X6 = X6.drop(['STR'], axis = 1)
X6 = X6.drop(['SCG'], axis = 1)
X6 = scale(X6)
df6 = pd.DataFrame(X6, columns = ["LPR", "PEG"])


result6 = KNeighborsClassifier(n_neighbors=5, weights = 'distance')


pred6 = cross_val_predict(result6, df6, y, cv=5)
a6 = accuracy_score(y, pred6)
p6 = precision_score(y, pred6, average='macro')
r6 = recall_score(y, pred6, average='macro')

print("drop str,scg and stg use scale to normalize 5 neightbors")
print("accuracy: "+ str(a6))
print("percision: "+ str(p6))
print("recall: "+ str(r6))
print("\n")

X7 = data.drop(['UNS'], axis = 1)
X7 = X7.drop(['STG'], axis = 1)
X7 = X7.drop(['STR'], axis = 1)
X7 = X7.drop(['SCG'], axis = 1)
X7 = scale(X7)
df7 = pd.DataFrame(X7, columns = ["LPR", "PEG"])


result7 = KNeighborsClassifier(n_neighbors=7, weights = 'distance')

pred7 = cross_val_predict(result7, df7, y, cv=5)
a7 = accuracy_score(y, pred7)
p7 = precision_score(y, pred7, average='macro')
r7 = recall_score(y, pred7, average='macro')

print("drop str,scg and stg use scale to normalize 7 neightbors")
print("accuracy: "+ str(a7))
print("percision: "+ str(p7))
print("recall: "+ str(r7))
print("\n")

X72 = data.drop(['UNS'], axis = 1)
X72 = X72.drop(['STG'], axis = 1)
X72 = X72.drop(['STR'], axis = 1)
X72 = X72.drop(['SCG'], axis = 1)
X72 = scale(X72)
df72 = pd.DataFrame(X72, columns = ["LPR", "PEG"])


result72 = KNeighborsClassifier(n_neighbors=7)


pred72 = cross_val_predict(result72, df72, y, cv=5)
a72 = accuracy_score(y, pred72)
p72 = precision_score(y, pred72, average='macro')
r72 = recall_score(y, pred72, average='macro')

print("drop str,scg and stg use scale to normalize 7 neightbors, weights at uniform")
print("accuracy: "+ str(a72))
print("percision: "+ str(p72))
print("recall: "+ str(r72))
print("\n")

'''
X = data.drop(['UNS'], axis = 1)
X = X.drop(['STG'], axis = 1)
X = X.drop(['STR'], axis = 1)
X = X.drop(['SCG'], axis = 1)
X = scale(X)
df = pd.DataFrame(X7, columns = ["LPR", "PEG"])

final = KNeighborsClassifier(n_neighbors=7, weights = 'distance')
final.fit(X, y)


tData = pd.read_csv("test.tsv", delimiter="\t")
alteredTest = tData.drop(['STG'], axis = 1)
alteredTest2 = alteredTest.drop(['STR'], axis = 1)
alteredTest3 = alteredTest2.drop(['SCG'], axis = 1)
testLabels = final.predict(alteredTest3)
'''

tData = pd.read_csv("test.tsv", delimiter="\t") 

Xf = data.drop(['UNS'], axis = 1)
Xf = Xf.drop(['STG'], axis = 1)
Xf = Xf.drop(['STR'], axis = 1)
Xf = Xf.drop(['SCG'], axis = 1)


final = KNeighborsClassifier(n_neighbors=7, weights = 'distance')
final.fit(Xf, y)

alteredTest = tData.drop(['STG'], axis = 1)
alteredTest2 = alteredTest.drop(['STR'], axis = 1)
alteredTest3 = alteredTest2.drop(['SCG'], axis = 1)
testLabels = final.predict(alteredTest3)

tData['label'] = testLabels
colors = {'Very_Low' : 'red', 'Low' : 'green', 'Middle':'purple', 'High':'cyan'}
y = tData['label']
c2 = [colors[x] for x in y]
pd.plotting.scatter_matrix(tData, c=c2)
plt.suptitle("red:verylow, low:green, middle:purple, high:cyan")
