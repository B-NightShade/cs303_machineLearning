# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:51:29 2022

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
plt.show()



X = data.drop(['UNS'], axis = 1)
X = scale(X)
df = pd.DataFrame(X, columns = ["STG", "SCG", "STR", "LPR", "PEG"])

allThree = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
allThree.fit(df, y)
result = KNeighborsClassifier()


pred = cross_val_predict(result, df, y, cv=5)
a = accuracy_score(y, pred)
p = precision_score(y, pred, average='micro')
r = recall_score(y, pred, average='micro')

print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

X2 = data.drop(['UNS'], axis = 1)
X2 = X2.drop(['STG'], axis = 1)
X2 = scale(X2)
df2 = pd.DataFrame(X2, columns = ["SCG", "STR", "LPR", "PEG"])

dropSTG = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
dropSTG.fit(df2, y)
result2 = KNeighborsClassifier()


pred2 = cross_val_predict(result2, df2, y, cv=5)
a2 = accuracy_score(y, pred2)
p2 = precision_score(y, pred2, average='micro')
r2 = recall_score(y, pred2, average='micro')

print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

X3 = data.drop(['UNS'], axis = 1)
X3 = X3.drop(['STR'], axis = 1)
X3 = scale(X3)
df3 = pd.DataFrame(X3, columns = ["STG","SCG","LPR", "PEG"])

dropSTR = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
dropSTR.fit(df3, y)
result3 = KNeighborsClassifier()


pred3 = cross_val_predict(result3, df3, y, cv=5)
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
X4 = scale(X4)
df4 = pd.DataFrame(X4, columns = ["SCG", "LPR", "PEG"])

drop2 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
drop2.fit(df4, y)
result4 = KNeighborsClassifier()


pred4 = cross_val_predict(result4, df4, y, cv=5)
a4 = accuracy_score(y, pred4)
p4 = precision_score(y, pred4, average='micro')
r4 = recall_score(y, pred4, average='micro')

print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")

X5 = data.drop(['UNS'], axis = 1)
X5 = X5.drop(['STG'], axis = 1)
X5 = X5.drop(['STR'], axis = 1)
X5 = X5.drop(['SCG'], axis = 1)
df5 = pd.DataFrame(X5, columns = ["LPR", "PEG"])

drop3 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
drop3.fit(df5, y)
result5 = KNeighborsClassifier()


pred5 = cross_val_predict(result5, df5, y, cv=5)
a5 = accuracy_score(y, pred5)
p5 = precision_score(y, pred5, average='micro')
r5 = recall_score(y, pred5, average='micro')

print("accuracy: "+ str(a5))
print("percision: "+ str(p5))
print("recall: "+ str(r5))
print("\n")

X5 = data.drop(['UNS'], axis = 1)
X5 = X5.drop(['STG'], axis = 1)
X5 = X5.drop(['STR'], axis = 1)
X5 = X5.drop(['SCG'], axis = 1)
df5 = pd.DataFrame(X5, columns = ["LPR", "PEG"])

drop3 = KNeighborsClassifier(n_neighbors=3, weights = 'distance', algorithm = 'ball_tree')
drop3.fit(df5, y)
result5 = KNeighborsClassifier()


pred5 = cross_val_predict(result5, df5, y, cv=5)
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
df6 = pd.DataFrame(X6, columns = ["LPR", "PEG"])

drop = KNeighborsClassifier(n_neighbors=7, weights = 'distance')
drop.fit(df6, y)
result6 = KNeighborsClassifier()


pred6 = cross_val_predict(result6, df6, y, cv=5)
a6 = accuracy_score(y, pred6)
p6 = precision_score(y, pred6, average='micro')
r6 = recall_score(y, pred6, average='micro')

print("accuracy: "+ str(a6))
print("percision: "+ str(p6))
print("recall: "+ str(r6))
print("\n")

accuracy=[]
for i in range(1, 15):
    X8 = data.drop(['UNS'], axis = 1)
    X8 = X8.drop(['STG'], axis = 1)
    X8 = X8.drop(['STR'], axis = 1)
    X8 = X8.drop(['SCG'], axis = 1)
    df8 = pd.DataFrame(X8, columns = ["LPR", "PEG"])
    result8 = KNeighborsClassifier(n_neighbors=i)
    pred8 = cross_val_predict(result8, df8, y, cv=5)
    a8 = accuracy_score(y, pred8)
    accuracy.append(a8)
plt.plot(range(1,15), accuracy, color = 'purple')
plt.xlabel("number neighbors")
plt.ylabel("accuracy")
plt.title("# neighbors vs accuracy LPR & PEG")
plt.show()

percision=[]
for i in range(1, 15):
    X8 = data.drop(['UNS'], axis = 1)
    X8 = X8.drop(['STG'], axis = 1)
    X8 = X8.drop(['STR'], axis = 1)
    X8 = X8.drop(['SCG'], axis = 1)
    df8 = pd.DataFrame(X8, columns = ["LPR", "PEG"])
    result8 = KNeighborsClassifier(n_neighbors=i)
    pred8 = cross_val_predict(result8, df8, y, cv=5)
    p8 = recall_score(y, pred8, average='macro')
    percision.append(p8)
plt.plot(range(1,15), percision, color = 'red')
plt.xlabel("number neighbors")
plt.ylabel("percision")
plt.title("# neighbors vs percision LPR & PEG")
plt.show()

recall=[]
for i in range(1, 15):
    X8 = data.drop(['UNS'], axis = 1)
    X8 = X8.drop(['STG'], axis = 1)
    X8 = X8.drop(['STR'], axis = 1)
    X8 = X8.drop(['SCG'], axis = 1)
    df8 = pd.DataFrame(X8, columns = ["LPR", "PEG"])
    result8 = KNeighborsClassifier(n_neighbors=i)
    pred8 = cross_val_predict(result8, df8, y, cv=5)
    r8 = precision_score(y, pred8, average='macro')
    recall.append(r8)
plt.plot(range(1,15), recall, color = 'gray')
plt.xlabel("number neighbors")
plt.ylabel("recall")
plt.title("# neighbors vs recall LPR & PEG")
plt.show()


recall=[]
percision=[]


tData = pd.read_csv("test.tsv", delimiter="\t")
'''
final = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
Xfinal = data.drop(['UNS'], axis = 1)
Xfinal = Xfinal.drop(['STG'], axis = 1)
Xfinal = Xfinal.drop(['STR'], axis = 1)
Xfinal = Xfinal.drop(['SCG'], axis = 1)
Xfinal = scale(Xfinal)
final.fit(Xfinal, y)

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
'''
tData = pd.read_csv("test.tsv", delimiter="\t")
final = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
final.fit(X5, y)
alteredTest = tData.drop(['STG'], axis = 1)
alteredTest2 = alteredTest.drop(['STR'], axis = 1)
alteredTest3 = alteredTest2.drop(['SCG'], axis = 1)
testLabels = final.predict(alteredTest3)

tData['label'] = testLabels
colors = {'Very_Low' : 'red', 'Low' : 'green', 'Middle':'purple', 'High':'cyan'}
y = tData['label']
c2 = [colors[x] for x in y]
pd.plotting.scatter_matrix(tData, c=c2)

