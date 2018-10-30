# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/24
    Desc : 
    Note : 
'''

import matplotlib.pyplot as plot
import pandas as panda

# opening datafile

datafile = panda.read_csv('./datas/Customers.csv')
X = datafile.iloc[:, [3, 4]].values

# applying K-Means to  datafile

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 10)
y_kmeans = kmeans.fit_predict(X)

# Visualisation of clusters

plot.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'magenta', label = 'Cluster 1')
plot.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plot.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plot.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plot.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'blue', label = 'Cluster 5')
plot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plot.title('Clusters of customers')
plot.xlabel('Annual Income (k$)')
plot.ylabel('Spending Score (1-100)')
plot.legend()
plot.show()