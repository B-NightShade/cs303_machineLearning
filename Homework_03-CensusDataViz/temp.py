# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column_names = ["age", "workclass", "fnlwg", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data = pd.read_csv('adult.data', header = None, delimiter = ', ', names = column_names)

'''
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
categories = ['highFincome', 'highMincome', 'lowFincome', 'lowMincome']
vals = [perHf, perHm, perlf, perlm]
plt.title("sex vs income")
plt.xlabel("category of income")
plt.ylabel("percent in that group")
colors =['black', 'red', 'cyan', 'green']
plt.bar(categories, vals, color = colors)
plt.legend()
plt.show()

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


white =  data['race']== 'White'
acp = data['race']== 'Asian-Pac-Islander'
aie = data['race']== 'Amer-Indian-Eskimo'
black = data['race'] == 'Black'
other = data['race'] == 'Other'

cgWhite = data['education-num'][white]
cgACP = data['education-num'][acp]
cgAIE = data['education-num'][aie]
cgBLACK = data['education-num'][black]
cgOTHER = data['education-num'][other]

plt.figure()

plt.hist(cgWhite, alpha = 0.45, label ='white')
plt.hist(cgACP, alpha = 0.45, label ='Asian-Pac-Islander')
plt.hist(cgAIE, alpha = 0.45, label ='Amer-Indian-Eskimo')
plt.hist(cgBLACK, alpha = 0.45, label ='black')
plt.hist(cgOTHER, alpha = 0.45, label ='other')
plt.legend()

plt.show()

plt.figure()
plt.title('education num and race')
plt.boxplot([cgWhite, cgACP, cgAIE, cgBLACK, cgOTHER])
labelg = ['white', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'black', 'other']
plt.xticks(ticks=[1,2,3,4,5], labels = labelg, rotation=45)
plt.xlabel("race")
plt.ylabel("education number")
plt.show()

'
plt.figure()

men =  data['sex']== 'Male'
women = data['sex']== 'Female'

mED = data['education-num'][men]
wED = data['education-num'][women]


plt.hist(mED, alpha = 0.45, bins = 16, label ='men')
plt.hist(wED, alpha = 0.45, bins = 16, label ='women')
plt.legend()

plt.show()



whiteH = np.where((data['income']== '>50K') & (data['race']=='White'))

WHtotal = 0
values, counts =np.unique(whiteH, return_counts=True)
for count in counts:
    WHtotal += int(count)

whiteL = np.where((data['income']== '<=50K') & (data['race']=='White'))

WLtotal = 0
values, counts =np.unique(whiteL, return_counts=True)
for count in counts:
    WLtotal += int(count)
    
apiH = np.where((data['income']== '>50K') & (data['race']=='Asian-Pac-Islander'))

APItotal = 0
values, counts =np.unique(apiH, return_counts=True)
for count in counts:
    APItotal += int(count)

apiL = np.where((data['income']== '<=50K') & (data['race']=='Asian-Pac-Islander'))

APILtotal = 0
values, counts =np.unique(apiL, return_counts=True)
for count in counts:
    APILtotal += int(count)
    
aieH = np.where((data['income']== '>50K') & (data['race']=='Amer-Indian-Eskimo'))

aietotal = 0
values, counts =np.unique(aieH, return_counts=True)
for count in counts:
    aietotal += int(count)

aieL = np.where((data['income']== '<=50K') & (data['race']=='Amer-Indian-Eskimo'))

aieLtotal = 0
values, counts =np.unique(aieL, return_counts=True)
for count in counts:
    aieLtotal += int(count)


bH = np.where((data['income']== '>50K') & (data['race']=='Black'))

btotal = 0
values, counts =np.unique(bH, return_counts=True)
for count in counts:
    btotal += int(count)

bL = np.where((data['income']== '<=50K') & (data['race']=='Black'))

bLtotal = 0
values, counts =np.unique(bL, return_counts=True)
for count in counts:
    bLtotal += int(count)
    
oH = np.where((data['income']== '>50K') & (data['race']=='Other'))

ototal = 0
values, counts =np.unique(oH, return_counts=True)
for count in counts:
    ototal += int(count)

oL = np.where((data['income']== '<=50K') & (data['race']=='Other'))

oLtotal = 0
values, counts =np.unique(oL, return_counts=True)
for count in counts:
    oLtotal += int(count)

whiteHI = WHtotal/(WHtotal+WLtotal)
whiteLI = WLtotal/(WHtotal+WLtotal)

apiHI = APItotal/(APItotal+APILtotal)
apiLI = APILtotal/(APItotal+APILtotal)

aieHI = aietotal/(aietotal+aieLtotal)
aieLI = aieLtotal/(aietotal+aieLtotal)

bHI = btotal/(btotal+bLtotal)
bLI = bLtotal/(btotal+bLtotal)

oHI = ototal/(ototal+oLtotal)
oLI = oLtotal/(ototal+oLtotal)

plt.figure()
plt.bar("white hi",whiteHI, color ='black', label = ">50K income")
plt.bar("white li",whiteLI, color ='red', label = "<=50K income")
plt.bar('Asian-Pac-Islander hi',apiHI, color ='black')
plt.bar("Asian-Pac-Islander li",apiLI, color ='red')
plt.bar('Amer-Indian-Eskimo hi',aieHI, color ='black')
plt.bar('Amer-Indian-Eskimo li',aieLI, color ='red')
plt.bar("black hi",bHI, color ='black')
plt.bar("black li",bLI, color ='red')
plt.bar("other hi",oHI, color ='black')
plt.bar("other li",oLI, color ='red')
plt.title("Race and income")
plt.ylabel("percent in category")
plt.xlabel("race category")
plt.xticks(rotation=90)
plt.legend()
plt.show()
'''

private = data['workclass'] == 'Private'
selfEmpNi = data['workclass'] == 'Self-emp-not-inc'
selfEmp = data['workclass']== 'Self-emp-inc'
fedGov = data['workclass']== 'Federal-gov'
locGov = data['workclass'] == 'Local-gov'
stateGov = data['workclass'] == 'State-gov'
withoutPay = data['workclass'] == 'Without-pay'
neverWorked = data['workclass'] == 'Never-worked'

apriv = data['age'][private]
aselfEmpNi = data['age'][selfEmpNi]
aselfEmp = data['age'][selfEmp]
afedG = data['age'][fedGov]
alocG = data['age'][locGov]
astaG = data['age'][stateGov]
aWP = data['age'][withoutPay]
aNW = data['age'][neverWorked]

plt.figure()
plt.title('age and workclass')
plt.boxplot([apriv, aselfEmpNi, aselfEmp, afedG, alocG, astaG, aWP, aNW], showmeans=True)
labelg = ['private', 'Self-emp-not-inc', 'Self-emp-inc' , 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
plt.xticks(ticks=[1,2,3,4,5,6,7,8], labels = labelg, rotation=45)
plt.xlabel("workclass")
plt.ylabel("age")
plt.show()






