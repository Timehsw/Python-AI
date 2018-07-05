# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 特征选择&降维
'''

import numpy as np
import warnings

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier

X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0.1, 1, 1, 3]
], dtype=np.float64)
Y = np.array([1,2,1])

print('-'*40,"方差选择法")
# 方差选择法
# threshold各个特征属性的阈值,获取方差大于阈值的特征
variance = VarianceThreshold(threshold=0.1)
print(variance)
variance.fit(X)
print(variance.transform(X))

print('-'*40,"相关系数法")
# 相关系数法
# 计算各个特征属性对于目标值的相关系数以及阈值K,然后获取K个相关系数最大的特征属性
sk1 = SelectKBest(f_regression, k=2)
sk1.fit(X,Y)
print(sk1.scores_)
print(sk1.transform(X))

print('-'*40,"卡方检验")
# 卡方检验
# 检查定性自变量对定性因变量的相关性
sk2 = SelectKBest(chi2, k=2)
sk2.fit(X, Y)
print(sk2)
print(sk2.scores_)
print(sk2.transform(X))

print('-'*40,"递归特征消除法")
# 递归特征消除法
# 使用一个基模型来进行多轮训练,每轮训练后,消除若干权值系数的特征,再基于新的特征集进行下一轮训练

estimator = SVR(kernel='linear')
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)
print(selector.transform(X))

print("~"*20)

X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = LogisticRegression(penalty='l1', C=0.1)
sfm = SelectFromModel(estimator)
sfm.fit(X2, Y2)
print(sfm.transform(X2))

print("~"*20)

X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2],
    [ 4.9,  3. ,  1.4,  0.2],
    [ -6.2,  0.4,  5.4,  2.3],
    [ -5.9,  0. ,  5.1,  1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = GradientBoostingClassifier()
sfm = SelectFromModel(estimator)
sfm.fit(X2, Y2)
print(sfm.transform(X2))

print('-'*40,"PCA")

from sklearn.decomposition import PCA
X2 = np.array([
    [ 5.1,  3.5,  1.4,  0.2, 1, 23],
    [ 4.9,  3. ,  1.4,  0.2, 2.3, 2.1],
    [ -6.2,  0.4,  5.4,  2.3, 2, 23],
    [ -5.9,  0. ,  5.1,  1.8, 2, 3]
], dtype=np.float64)
# n_components: 给定降低到多少维度，但是要求该值必须小于等于样本数目/特征数目，如果给定的值大于，那么会选择样本数目/特征数目中最小的那个作为最终的特征数目
pca = PCA(n_components=3)
pca.fit(X2)
print(pca.mean_)
print(pca.components_)
print(pca.transform(X2))

print("~"*20)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.array([
    [-1, -1, 3, 1],
    [-2, -1, 2, 4],
    [-3, -2, 4, 5],
    [1, 1, 5, 4],
    [2, 1, 6, -5],
    [3, 2, 1, 5]])
y = np.array([1, 1, 2, 2, 0, 1])
# n_components：给定降低到多少维度，要求给定的这个值和y的取值数量有关，不能超过n_class-1
clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(X, y)
print(clf.transform(X))