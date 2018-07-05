# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/7
    Desc : 层次聚类(AGNES)算法采用不同距离计算策略导致的数据合并不同形式
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import sklearn.datasets as ds
import warnings

# 设置属性防止中文乱码及拦截异常信息
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(action='ignore', category=UserWarning)

# 模拟数据产生:产生600条数据
np.random.seed(0)
n_cluster = 4
N = 1000
centers = [[-1, 1], [1, 1], [1, -1], [-1, -1]]
data1, y1 = ds.make_blobs(n_samples=N, n_features=2, centers=centers, random_state=0)

n_noise = int(0.1 * N)
r = np.random.rand(n_noise, 2)
min1, min2 = np.min(data1, axis=0)
max1, max2 = np.max(data1, axis=0)

r[:, 0] = r[:, 0] * (max1 - min1) + min1
r[:, 1] = r[:, 1] * (max2 - min2) + min2

# np.concatenate 对data1,和r进行拼接合并
data1_noise = np.concatenate((data1, r), axis=0)
y1_noise = np.concatenate((y1, [4] * n_noise))

#############################################################
# 拟合月牙形数据
data2, y2 = ds.make_moons(n_samples=N, noise=.05)
data2 = np.array(data2)
n_noise = int(0.1 * N)
r = np.random.rand(n_noise, 2)

min1, min2 = np.min(data2, axis=0)
max1, max2 = np.max(data2, axis=0)

r[:, 0] = r[:, 0] * (max1 - min1) + min1
r[:, 1] = r[:, 1] * (max2 - min2) + min2

data2_noise = np.concatenate((data2, r), axis=0)
y2_noise = np.concatenate((y2, [4] * n_noise))


def expandBorder(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


# 画图
# 给定画图的颜色
cm = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#d8e507', '#F0F0F0'])
plt.figure(figsize=(14, 12), facecolor='w')
linkages = ("ward", 'complete', 'average')  # 把几种距离方法,放到List里,后面直接循环取值

for index, (n_cluster, data, y) in enumerate(((4, data1, y1), (4, data1_noise, y1_noise),
                                              (2, data2, y2), (2, data2_noise, y2_noise))):
    plt.subplot(4, 4, 4 * index + 1)
    plt.scatter(data[:, 0], data[:, 1], c=y, cmap=cm)
    plt.title("原始数据", fontsize=17)
    plt.grid(b=True, ls=':')
    min1, min2 = np.min(data, axis=0)
    max1, max2 = np.max(data, axis=0)
    plt.xlim(expandBorder(min1, max2))
    plt.ylim(expandBorder(min2, max2))

    # 计算类别与类别的距离(只计算最接近的七个样本的距离)
    connectivity = kneighbors_graph(data, n_neighbors=7, mode='distance', metric='minkowski', p=2, include_self=True)
    connectivity = connectivity + connectivity.T

    for i, linkage in enumerate(linkages):
        # 进行建模,并传值
        # print(n_cluster)
        ac = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', connectivity=connectivity,
                                     linkage=linkage)
        ac.fit(data)
        y = ac.labels_

        plt.subplot(4, 4, i + 2 + 4 * index)
        plt.scatter(data[:, 0], data[:, 1], c=y, cmap=cm)
        plt.title(linkage, fontsize=17)
        plt.grid(b=True, ls=':')
        plt.xlim(expandBorder(min1, max1))
        plt.ylim(expandBorder(min2, max2))

plt.suptitle('AGNES层次聚类的不同合并策略,fontsize=30')
plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
plt.show()
