# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/16
    Desc : 进行聚类算法的评估指标计算
    Note : CP指数,SP指数,DB,DVI
'''

# 导入包
from numpy import *
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
from sklearn.cluster import KMeans  # 导入kmeans 类


def distEclud(vecA, vecB, axis=None):
    '''
    求两个向量之间的距离,计算欧几里得距离
    same as : np.linalg.norm(np.asarray(datas[0].iloc[0].values) - np.asarray(datas[0].iloc[1].values))
    :param vecA: 中心点
    :param vecB: 数据点
    :param axis: np.sum的参数
    :return:
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2), axis=axis))


def calculate_cp_i(cluster_center, cluster_point):
    '''
    计算聚类类各点到聚类中心的平均距离
    :param cluster_center: 聚类中心点
    :param cluster_point: 聚类样本点
    :return:
    '''
    return np.average(distEclud(cluster_center, cluster_point, axis=1))


def calculate_cp(cluster_centers, cluster_points):
    '''
    CP,计算每一个类各点到聚类中心的平均距离
    CP越低意味着类内聚类距离越近
    :param cluster_centers: 聚类中心点
    :param cluster_points: 聚类样本点
    :return:
    '''

    cps = [calculate_cp_i(cluster_centers[i], cluster_points[i]) for i in arange(len(cluster_centers))]
    cp = np.average(cps)
    return cp


def calculate_sp(cluster_centers):
    '''
    SP,计算各聚类中心两两之间的平均距离
    SP越高意味类间聚类距离越远
    :param cluster_centers:
    :return:
    '''
    k = len(cluster_centers)
    res = []
    for i in arange(k):
        tmp = []
        for j in arange(i + 1, k):
            tmp.append(distEclud(cluster_centers[i], cluster_centers[j]))
        res.append(np.sum(tmp))

    sp = (2 / (k * k - k)) * np.sum(res)
    return sp


def calculate_db(cluster_centers, cluster_points):
    '''
    DB,计算任意两类别的类内距离平均距离(CP)之和除以两聚类中心距离求最大值
    DB越小意味着类内距离越小 同时类间距离越大
    :param cluster_centers: 聚类中心点
    :param cluster_points: 聚类样本点
    :return:
    '''
    n_cluster = len(cluster_centers)
    cps = [calculate_cp_i(cluster_centers[i], cluster_points[i]) for i in arange(n_cluster)]
    db = []

    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((cps[i] + cps[j]) / distEclud(cluster_centers[i], cluster_centers[j]))
    db = (np.max(db) / n_cluster)
    return db


def calculate_dvi(cluster_points):
    '''
    DVI计算,任意两个簇元素的最短距离(类间)除以任意簇中的最大距离(类内).
    DVI越大意味着类间距离越大 同时类内距离越小。
    :param cluster_points:
    :return:
    '''
    # calcuation of maximum distance
    d1 = []
    d2 = []
    d3 = []
    # 类内最大距离
    for k, cluster_point in cluster_points.items():
        # 遍历每一个簇中每一个元素
        for i in range(len(cluster_point)):
            temp1 = cluster_point.iloc[i].values
            for j in range(len(cluster_point)):
                temp2 = cluster_point.iloc[j].values

                # 求簇内两元素的距离
                dist = distEclud(temp1, temp2)
                d1.insert(j, dist)

            d2.insert(i, max(d1))
            d1 = []

        d3.insert(k, max(d2))
        d2 = []
    xmax = max(d3)

    # calcuation of minimun distance
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    # 类间最小距离
    for k, cluster_point in cluster_points.items():
        # 遍历每一个簇中每一个元素
        for j in range(len(cluster_point)):
            temp1 = cluster_point.iloc[j].values
            for key in cluster_points.keys():
                if not key == k:
                    other_cluster_df = cluster_points[key]
                    for index in range(len(other_cluster_df)):
                        temp2 = other_cluster_df.iloc[index].values
                        dist = distEclud(temp1, temp2)
                        d1.insert(index, dist)

                    d2.insert(key, min(d1))
                    d1 = []
            d3.insert(j, min(d2))
            d2 = []
        d4.insert(k, min(d3))
    xmin = min(d4)

    dunn_index = xmin / xmax
    return dunn_index


if __name__ == '__main__':
    # 1. 产生模拟数据
    N = 1000
    centers = 4
    X, Y = make_blobs(n_samples=N, n_features=2, centers=centers, random_state=28)

    # 2. 模型构建
    km = KMeans(n_clusters=centers, init='random', random_state=28)
    km.fit(X)

    # 模型的预测
    y_hat = km.predict(X[:10])
    print(y_hat)

    print("所有样本距离所属簇中心点的总距离和为:%.5f" % km.inertia_)
    print("所有样本距离所属簇中心点的平均距离为:%.5f" % (km.inertia_ / N))

    print("所有的中心点聚类中心坐标:")
    cluster_centers = km.cluster_centers_
    print(cluster_centers)

    print("score其实就是所有样本点离所属簇中心点距离和的相反数:")
    print(km.score(X))

    # 统计各个聚簇点的类别数目
    r1 = pd.Series(km.labels_).value_counts()

    kmlabels = km.labels_  # 得到类别，label_ 是内部变量
    X = pd.DataFrame(X, columns=list('ab'))
    r = pd.concat([X, pd.Series(kmlabels, index=X.index)], axis=1)  # 详细输出每个样本对应的类别，横向连接（0是纵向），得到聚类中心对应的类别下的数目

    r.columns = list(X.columns) + ['class']  # 重命名表头  加一列的表头
    print(r.head())

    # 根据聚类信息,将数据进行分类{聚类id:聚类点}
    cluster_centers = {k: value for k, value in enumerate(cluster_centers)}
    cluster_points = {label: r[r['class'] == label].drop(columns=['class']) for label in np.unique(km.labels_)}

    print('~' * 10, 'evaluate kmeans', '~' * 10)

    cp = calculate_cp(cluster_centers, cluster_points)
    print('cp : ', cp)

    sp = calculate_sp(cluster_centers)
    print('sp : ', sp)

    dp = calculate_db(cluster_centers, cluster_points)
    print('dp : ', dp)

    dvi = calculate_dvi(cluster_points)
    print('dvi : ', dvi)
