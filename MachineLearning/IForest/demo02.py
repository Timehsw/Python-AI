# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/15
    Desc : 孤立森林例子
    Note : 
'''

import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# 生成训练数据
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 1, X - 3, X - 5, X + 6]

# 生成正常数据
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 1, X - 3, X - 5, X + 6]

# 生成异常数据
X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))

'''
算法主要有两个参数，一个是二叉树的个数，另一个是训练单棵iTree时候抽取样本的数目

实验表明，当设定参数为 100 棵树，抽样样本数为 256 条时候，在大多数情况下就已经可以取得不错的效果。
'''

# 使用模型
clf = IsolationForest(max_samples=100 * 2, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# 统计预测出来的训练集,测试集等里面有哪些是异常点
# 异常点是-1
np.unique(y_pred_train,return_counts=True)
np.unique(y_pred_test,return_counts=True)
np.unique(y_pred_outliers,return_counts=True)

# 返回是异常点的数据索引
np.where(y_pred_train==-1)

# 拿出训练集中的异常点
X_train[np.where(y_pred_train==-1)]