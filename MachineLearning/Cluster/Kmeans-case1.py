# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/4
    Desc : 
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 产生模拟数据
N = 150
centers = 4
data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=28)

# 模型构建
km = KMeans(n_clusters=centers, init='random', random_state=28)
# init 初始化方法,可以是kmeans++,随机,或者自定义的ndarry
km.fit(data, y)

# 模型预测
y_hat = km.predict(data)
print("所有样本距离聚簇中心点的总距离和:", km.inertia_)
print("距离聚簇中心点的平均距离:", (km.cluster_centers_ / N))
print("聚簇中心点:", km.cluster_centers_)


def expandBorder(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


# 画图
cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15, 9), facecolor='w')
plt.subplot(241)
plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')

x1_min, x2_min = np.min(data, axis=0)
x1_max, x2_max = np.max(data, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('原始数据')
plt.grid(True)

plt.subplot(242)
plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title("k-means算法聚类结果")
plt.grid(True)

plt.show()
