# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:48:05 2022

@author: weste
"""

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy

names = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
data = pd.read_csv("mystery.csv", delimiter = "\t", names = names)

'''
for i in range(2,8):
    plt.plot(data['c1'], data['c'+str(i)], 'b*')
    plt.xlabel("c1")
    plt.ylabel("c"+str(i))
    plt.title("plot c1 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c1"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,0], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c1")
    plt.title("plot c1 and c" +str(i))
    plt.show()

 
for i in range(3,8):
    plt.plot(data['c2'], data['c'+str(i)], 'g*')
    plt.xlabel("c2")
    plt.ylabel("c"+str(i))
    plt.title("plot c2 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c2"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,1], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c2")
    plt.title("plot c2 and c" +str(i))
    plt.show()


for i in range(4,8):
    plt.plot(data['c3'], data['c'+str(i)], 'r*')
    plt.xlabel("c3")
    plt.ylabel("c"+str(i))
    plt.title("plot c3 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c3"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,2], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c3")
    plt.title("plot c3 and c" +str(i))
    plt.show()

for i in range(5,8):
    plt.plot(data['c4'], data['c'+str(i)], 'c*')
    plt.xlabel("c4")
    plt.ylabel("c"+str(i))
    plt.title("plot c4 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c4"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,3], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c4")
    plt.title("plot c4 and c" +str(i))
    plt.show()

for i in range(6,8):
    plt.plot(data['c5'], data['c'+str(i)], 'm*')
    plt.xlabel("c5")
    plt.ylabel("c"+str(i))
    plt.title("plot c5 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c5"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,4], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c5")
    plt.title("plot c5 and c" +str(i))
    plt.show()
    
for i in range(7,8):
    plt.plot(data['c6'], data['c'+str(i)], 'y*')
    plt.xlabel("c6")
    plt.ylabel("c"+str(i))
    plt.title("plot c6 and c" +str(i))
    plt.show()
    kmeans = KMeans(n_clusters = 3).fit(data)
    clusterLabels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data["c6"], data["c"+str(i)], c = clusterLabels, cmap='prism')
    plt.plot(centroids[:,5], centroids[:,i-1], 'k+', markersize = 12)
    plt.ylabel("c"+str(i))
    plt.xlabel("c6")
    plt.title("plot c6 and c" +str(i))
    plt.show()

    dendro = hierarchy.dendrogram(hierarchy.linkage(data, method='average'), color_threshold=(0))
    plt.xticks(rotation = 90)
    plt.show()
'''
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.plot(reduced["column1"], reduced["column2"], 'b*')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("PCA plot 2d")
plt.show()

kmeans = KMeans(n_clusters = 2)
result = kmeans.fit_predict(data)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = result, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("2 k clusters w/ PCA 2")
plt.show()

pd.plotting.scatter_matrix(data, c = result);
plt.show()

kmeans = KMeans(n_clusters = 3)
result = kmeans.fit_predict(data)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = result, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("3 k clusters w/ PCA 2")
plt.show()

ax = plt.axes(projection='3d')
dataPCA = PCA(n_components=(3)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2", "column3"])
plt.plot(reduced["column1"], reduced["column2"], 'b*')
ax.set_ylabel("reduced Column2")
ax.set_xlabel("reduced Column2")
ax.set_zlabel("reduced column3")
ax.set_title("PCA plot 3d")
plt.show()

ax = plt.axes(projection='3d')
kmeans = KMeans(n_clusters = 3)
result = kmeans.fit_predict(data)
dataPCA2 = PCA(n_components=(3)).fit_transform(data)
reduced = pd.DataFrame(dataPCA2, columns = ["column1", "column2", "column3"])
ax.scatter3D(reduced["column1"], reduced["column2"],reduced["column3"], c = result, cmap='prism')
ax.set_ylabel("reduced Column2")
ax.set_xlabel("reduced Column2")
ax.set_zlabel("reduced column3")
ax.set_title("3 k clusters w/ PCA 3")
plt.show()

hier = AgglomerativeClustering(n_clusters=2, linkage = 'average')
model = hier.fit_predict(data)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = model, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("2 heirarchal clusters w/ PCA 2")
plt.show()

hier = AgglomerativeClustering(n_clusters=3, linkage = 'average')
model = hier.fit_predict(data)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = model, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("3 heirarchal clusters w/ PCA 2")
plt.show()

dbscan = DBSCAN()
model2 = dbscan.fit_predict(data)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = model2, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("db clusters w/ PCA 2")
plt.show()

ax = plt.axes(projection='3d')
dbscan = DBSCAN()
model = dbscan.fit_predict(data)
dataPCA = PCA(n_components=(3)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2", "column3"])
ax.scatter3D(reduced["column1"], reduced["column2"],reduced["column3"], c = model, cmap='hsv')
ax.set_ylabel("reduced Column2")
ax.set_xlabel("reduced Column1")
ax.set_zlabel("reduced column3")
ax.set_title("PCA plot 3d db clusters")
plt.show()


##iv refers to inertia an attribute of kMeans
##inertia is the sum of squared distances of samples to their closest cluster
iv=[]
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit_predict(data)
    iv.append(kmeans.inertia_)
plt.plot(range(1,10), iv, color = 'purple')
plt.xlabel("number clusters")
plt.ylabel("total Squared error")
plt.title("total error vs # clusters")
plt.show()


kmeans = KMeans(n_clusters = 3)
resultClusters = kmeans.fit_predict(data)
print(resultClusters)
dataPCA = PCA(n_components=(2)).fit_transform(data)
reduced = pd.DataFrame(dataPCA, columns = ["column1", "column2"])
plt.scatter(reduced["column1"], reduced["column2"], c = resultClusters, cmap='prism')
plt.ylabel("reduced Column2")
plt.xlabel("reduced Column1")
plt.title("3 k clusters w/ PCA 2")
plt.show()

pd.plotting.scatter_matrix(data, c = resultClusters)

data['label']=resultClusters

data.to_csv('Cumro.csv', header=False, index=False, sep='\t')

data2 = pd.read_csv("Cumro.csv", delimiter = "\t")
print(data2)

