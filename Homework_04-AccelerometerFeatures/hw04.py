# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:56:04 2022

@author: weste
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

'''
names = [ "i", "x", "y", "z", "activity"]

data1 = pd.read_csv("1.csv", delimiter=',', names=names)
data2 = pd.read_csv("2.csv", delimiter=',',names=names)
data3 = pd.read_csv("3.csv", delimiter=',', names=names)
data4 = pd.read_csv("4.csv", delimiter=',', names=names)
data5 = pd.read_csv("5.csv", delimiter=',', names=names)
data6 = pd.read_csv("6.csv", delimiter=',', names=names)
data7 = pd.read_csv("7.csv", delimiter=',', names=names)
data8 = pd.read_csv("8.csv", delimiter=',', names=names)
data9 = pd.read_csv("9.csv", delimiter=',', names=names)
data10 = pd.read_csv("10.csv", delimiter=',', names=names)
data11 = pd.read_csv("11.csv", delimiter=',', names=names)
data12 = pd.read_csv("12.csv", delimiter=',', names=names)
data13 = pd.read_csv("13.csv", delimiter=',', names=names)
data14 = pd.read_csv("14.csv", delimiter=',', names=names)
data15 = pd.read_csv("15.csv", delimiter=',', names=names)


A14 = data14.to_numpy()

wac14 = (A14[:,4]==1)
x = A14[wac14][:,[0,1]]
y = A14[wac14][:,[0,2]]
z = A14[wac14][:,[0,3]]

scaleX = scale(x[:,1])
scaleY = scale(y[:, 1])
scaleZ = scale(z[:, 1])


plt.plot((x[:,0])/52, scaleX+20)
plt.plot((y[:,0])/52, scaleY)
plt.plot((y[:,0])/52, scaleZ-20)
plt.xlabel("time(s)")
plt.ylabel("normalized magnitiude")
plt.title("Working at computer")
plt.legend(["x","y", "z"])
plt.xlim(left=0)

plt.show()


A14 = data14.to_numpy()

wac14 = (A14[:,4]==1)
x = A14[wac14][:,1]
y = A14[wac14][:,2]
z = A14[wac14][:,3]



scaleX = scale(x)
scaleY = scale(y)
scaleZ = scale(z)


plt.plot(scaleX+20)
plt.plot(scaleY)
plt.plot(scaleZ-20)
plt.xlabel("time(s)")
plt.ylabel("normalized magnitiude")
plt.title("Working at computer")
plt.legend(["x","y", "z"])
plt.xlim(left=0)

plt.show()
'''


participants = {}
names = [ "i", "x", "y", "z", "activity"]
for i in range(1,16):
    participants[i] = pd.read_csv(f"{i}.csv", header = None, names= names )

'''
activities = [" working @ computer", " SUWGUDS", " standing", " walking", " up/down stairs", " walk talk", " talk stand"]
for key in participants:
    data = participants[key].to_numpy()
    a = 1
    while(a<8):
        act = (data[:, 4] == a)
        x = data[act][:,[0,1]]
        y = data[act][:,[0,2]]
        z = data[act][:,[0,3]]
        scaleX = scale(x[:,1])
        scaleY = scale(y[:, 1])
        scaleZ = scale(z[:, 1])
        base = np.size(scaleX)
        sec = []
        for g in range(base):
            sec.append(g)
        secnp = np.array(sec)
        plt.plot(secnp/52,scaleX+20)
        plt.plot(secnp/52,scaleY)
        plt.plot(secnp/52,scaleZ-20)
        plt.xlabel("time(s)")
        plt.ylabel("normalized magnitiude")
        plt.legend(["x","y", "z"])
        plt.title("participant " + str(key) + activities[a-1])
        plt.xlim(left=0)
        plt.show()
        a+=1

for key in participants:
    if(key == 13 or key == 12 or key == 14):
        data = participants[key].to_numpy()
        act = (data[:, 4] == 4)
        x = data[act][:,[0,1]]
        y = data[act][:,[0,2]]
        z = data[act][:,[0,3]]
        scaleX = scale(x[:,1])
        scaleY = scale(y[:, 1])
        scaleZ = scale(z[:, 1])
        base = np.size(scaleX)
        sec = []
        for g in range(base):
            sec.append(g)
        secnp = np.array(sec)
        plt.plot(secnp/52,scaleX+20)
        plt.plot(secnp/52,scaleY)
        plt.plot(secnp/52,scaleZ-20)
        plt.xlabel("time(s)")
        plt.ylabel("normalized magnitiude")
        plt.legend(["x","y", "z"])
        plt.title("participant " + str(key) + " Walking")
        plt.xlim(left=0)
        plt.show()

for key in participants:
    data = participants[key].to_numpy()
    act = (data[:, 4] == 1)
    x = data[act][:,[0,1]]
    y = data[act][:,[0,2]]
    z = data[act][:,[0,3]]
    scaleX = scale(x[:,1])
    scaleY = scale(y[:, 1])
    scaleZ = scale(z[:, 1])
    base = np.size(scaleX)
    sec = []
    for g in range(base):
        sec.append(g)
    secnp = np.array(sec)
    plt.plot(secnp/52,scaleX+20)
    plt.plot(secnp/52,scaleY)
    plt.plot(secnp/52,scaleZ-20)
    plt.xlabel("time(s)")
    plt.ylabel("normalized magnitiude")
    plt.legend(["x","y", "z"])
    plt.title("participant " + str(key) + " working @ computer")
    plt.xlim(left=0)
    plt.show()
    
'''

for key in participants:
    if(key == 9 or key == 2 or key == 10 or key == 7):
        data = participants[key].to_numpy()
        act = (data[:, 4] == 3)
        x = data[act][:,[0,1]]
        y = data[act][:,[0,2]]
        z = data[act][:,[0,3]]
        scaleX = scale(x[:,1])
        scaleY = scale(y[:, 1])
        scaleZ = scale(z[:, 1])
        base = np.size(scaleX)
        sec = []
        for g in range(base):
            sec.append(g)
        secnp = np.array(sec)
        plt.plot(secnp/52,scaleX+20)
        plt.plot(secnp/52,scaleY)
        plt.plot(secnp/52,scaleZ-20)
        plt.xlabel("time(s)")
        plt.ylabel("normalized magnitiude")
        plt.legend(["x","y", "z"])
        plt.title("participant " + str(key) + " standing")
        plt.xlim(left=0)
        plt.show()





