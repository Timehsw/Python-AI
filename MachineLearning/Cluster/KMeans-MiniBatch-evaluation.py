# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/5
    Desc : K-Means 和 Mini Batch K-Means 效果评估
'''

import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

centers = [[1, 1], [-1, 1], [1, -1]]
cluster = len(centers)

x, y = make_blobs(n_samples=1000, centers=centers, cluster_std=0.7, random_state=28)
print(x)
# 在实际工作中是人工给定的，专门用于判断聚类的效果的一个值
### TODO: 实际工作中，我们假定聚类算法的模型都是比较可以，最多用轮廓系数/模型的score api返回值进行度量；
### 其它的效果度量方式一般不用
### 原因：其它度量方式需要给定数据的实际的y值 ===> 当我给定y值的时候，其实我可以直接使用分类算法了，不需要使用聚类

k_means = KMeans(init='k-means++', n_clusters=cluster, random_state=28)
t0 = time.time()
k_means.fit(x)
km_batch = time.time() - t0
print('K-Means算法模型训练消耗时间:%.4fs' % km_batch)

mbk = MiniBatchKMeans(init='k-means++', n_clusters=cluster, batch_size=100, random_state=28)
t0 = time.time()
mbk.fit(x)
mbk_batch = time.time() - t0
print('Mini Batch K-Means算法模型训练消耗时间:%.2fs' % (mbk_batch))

km_y_hat = k_means.labels_ # 样本所属类别,也就是y值
mbkm_y_hat = mbk.labels_
print(y[:20])
print(km_y_hat[:20])

k_means.cluster_centers_
