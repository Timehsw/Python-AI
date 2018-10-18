# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/16
    Desc : 
    Note : 
'''

# 导入包
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs # 导入产生模拟数据的方法
from sklearn.cluster import KMeans # 导入kmeans 类
import matplotlib.pyplot as plt  #matplotlib画图

# 1. 产生模拟数据
N = 1000
centers = 4
X, Y = make_blobs(n_samples=N, n_features=2, centers=centers, random_state=28)

# 2. 模型构建
km = KMeans(n_clusters=centers, init='random', random_state=28)
km.fit(X)

# 模型的预测
y_hat = km.predict(X[:10])
y_hat

print("所有样本距离所属簇中心点的总距离和为:%.5f" % km.inertia_)
print("所有样本距离所属簇中心点的平均距离为:%.5f" % (km.inertia_ / N))

print("所有的中心点聚类中心坐标:")
cluter_centers = km.cluster_centers_
print(cluter_centers)

print("score其实就是所有样本点离所属簇中心点距离和的相反数:")
print(km.score(X))

# 统计各个聚簇点的类别数目
r1=pd.Series(km.labels_).value_counts()

kmlabels = km.labels_  # 得到类别，label_ 是内部变量
X=pd.DataFrame(X,columns=list('ab'))
r = pd.concat([X, pd.Series(kmlabels, index=X.index)], axis=1)  # 详细输出每个样本对应的类别，横向连接（0是纵向），得到聚类中心对应的类别下的数目

r.columns = list(X.columns) + ['class']  # 重命名表头  加一列的表头
print(r.head())

mink = 4  #聚类的类别范围K值下界
maxk = mink + 1 #聚类的列别范围上界


def density_plot(data):  # 自定义作图函数
    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    [p[i].set_ylabel('density') for i in range(k)]
    plt.legend()
    return plt

if __name__ == '__main__':



    pic_output = '/Users/hushiwei/Downloads'  # 概率密度图文件名前缀
    k=len(kmlabels)
    for i in range(k):
        print(u'%s%s.png' % (pic_output, i))
        if i > 1:
            density_plot(X[r[u'class'] == i]).savefig(u'%s%s.png' % (pic_output, i))