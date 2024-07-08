# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:59:22 2022

@author: weste
"""
'''
accuracy=[]
for i in range(1, 25):
    dtc = DecisionTreeClassifier(max_depth=(i), splitter='random', random_state=(2))
    result= dtc.fit(dataHalf2,y)
    y_pred = cross_val_predict(result, dataHalf2, y, cv=5)
    a3 = accuracy_score(y, y_pred)
    accuracy.append(a3)
plt.plot(range(1,25), accuracy, color = 'purple')
plt.xlabel("max depth")
plt.ylabel("accuracy")
plt.title("# max depth vs accuracy")
plt.show()

percision=[]
for i in range(1, 25):
    dtc = DecisionTreeClassifier(max_depth=(i), splitter='random', random_state=(2))
    result= dtc.fit(dataHalf2,y)
    y_pred = cross_val_predict(result, dataHalf2, y, cv=5)
    p3 = precision_score(y, y_pred, average='macro')
    percision.append(p3)
plt.plot(range(1,25), percision, color = 'red')
plt.xlabel("max depth")
plt.ylabel("percision")
plt.title("# max depth percision")
plt.show()

recall=[]
for i in range(1, 25):
    dtc = DecisionTreeClassifier(max_depth=(i), splitter='random', random_state=(2))
    result= dtc.fit(dataHalf2,y)
    y_pred = cross_val_predict(result, dataHalf2, y, cv=5)
    r3 = recall_score(y, y_pred, average='macro')
    recall.append(r3)
plt.plot(range(1,25), recall, color = 'blue')
plt.xlabel("max depth")
plt.ylabel("recall")
plt.title("# max depth recall")
plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv", delimiter=(','))

class1 = data[data['Class']==1]
class0 = data[data['Class']==0]

for i in range (1,5):
    plt.subplot(2,2,i)
    apol1 = class1['Atr' + str(i)]
    apol0 = class0['Atr' + str(i)]
    plt.boxplot([apol1, apol0], labels= [1,0])
    plt.xlabel("class")
    plt.ylabel("score/value")
    plt.title("atr" + str(i))
plt.show()
   
x = 5
for i in range (0, 12):
    for i in range (1,5):
        plt.subplot(2,2,i)
        apol1 = class1['Atr' + str(x)]
        apol0 = class0['Atr' + str(x)]
        plt.boxplot([apol1, apol0], labels= [1,0])
        plt.xlabel("class")
        plt.ylabel("score/value")
        plt.title("atr" + str(x))
        x+=1
    plt.show()


plt.subplot(2,1,1)
apol1 = class1['Atr53']
apol0 = class0['Atr53']
plt.boxplot([apol1, apol0], labels= [1,0])
plt.xlabel("class")
plt.ylabel("score/value")
plt.title("atr53")
plt.subplot(2,1,2)
apol1 = class1['Atr54']
apol0 = class0['Atr54']
plt.boxplot([apol1, apol0], labels= [1,0])
plt.xlabel("class")
plt.ylabel("score/value")
plt.title("atr54")
plt.show()


y = data['Class']
X = data.drop(['Class'], axis = 1)
'''
correlations = (X.corr())
print(correlations)

HC = correlations > 0.87
print(HC)


dtm = DecisionTreeClassifier(random_state=(2))
dtm.fit(X,y)

y_pred = cross_val_predict(dtm, X, y, cv=5)
a = accuracy_score(y, y_pred)
p = precision_score(y, y_pred, average='macro')
r = recall_score(y, y_pred, average='macro')

print("all attributes-generic")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")


dtm = DecisionTreeClassifier(splitter='random', random_state=(2))
outcome = dtm.fit(X,y)

y_pred = cross_val_predict(outcome, X, y, cv=5)
a = accuracy_score(y, y_pred)
p = precision_score(y, y_pred, average='macro')
r = recall_score(y, y_pred, average='macro')

print("all attributes- split random")
print("accuracy: "+ str(a))
print("percision: "+ str(p))
print("recall: "+ str(r))
print("\n")

X2 = X.drop(['Atr5', 'Atr2', "Atr11", "Atr33", "Atr35"], axis = 1)
dtm2 = DecisionTreeClassifier(random_state=(2))
result= dtm2.fit(X2,y)

y_pred = cross_val_predict(result, X2, y, cv=5)
a2 = accuracy_score(y, y_pred)
p2 = precision_score(y, y_pred, average='macro')
r2 = recall_score(y, y_pred, average='macro')

print("all attributes-generic")
print("accuracy: "+ str(a2))
print("percision: "+ str(p2))
print("recall: "+ str(r2))
print("\n")

dtm3 = DecisionTreeClassifier(max_depth=(5), splitter='random', random_state=(2))
result= dtm3.fit(X,y)

y_pred = cross_val_predict(result, X, y, cv=5)
a4 = accuracy_score(y, y_pred)
p4 = precision_score(y, y_pred, average='macro')
r4 = recall_score(y, y_pred, average='macro')

print("all attributes- random depth 5")
print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")

dataHalf = X.iloc[:,:27]

dtm3 = DecisionTreeClassifier(max_depth=(5), splitter='random', random_state=(2))
result= dtm3.fit(dataHalf,y)

y_pred = cross_val_predict(result, dataHalf, y, cv=5)
a4 = accuracy_score(y, y_pred)
p4 = precision_score(y, y_pred, average='macro')
r4 = recall_score(y, y_pred, average='macro')

print("first half attributes- random -depth 5")
print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")


dataHalf2 = X.iloc[:,27:]

dtm3 = DecisionTreeClassifier(max_depth=(5), splitter='random' , random_state=(2))
result= dtm3.fit(dataHalf2,y)

y_pred = cross_val_predict(result, dataHalf2, y, cv=5)
a4 = accuracy_score(y, y_pred)
p4 = precision_score(y, y_pred, average='macro')
r4 = recall_score(y, y_pred, average='macro')

print("2nd half attributes- random -depth 5")
print("accuracy: "+ str(a4))
print("percision: "+ str(p4))
print("recall: "+ str(r4))
print("\n")
'''
testData = pd.read_csv("test.csv", delimiter=",")

Xhalf2 = X.iloc[:,27:]
finalTree = DecisionTreeClassifier(max_depth=(5), splitter='random' , random_state=(2))
finalresult = finalTree.fit(Xhalf2,y)

alteredTest = testData.iloc[:,27:]
testLabels = finalresult.predict(alteredTest)

testData['Class'] = testLabels