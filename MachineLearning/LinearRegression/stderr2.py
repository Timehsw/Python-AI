# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/8
    Desc : 
    Note : 
'''

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc

# 加载数据
path = './datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
# print(df.head())
# print(df.shape)
datas = df.replace('?', np.nan).dropna(how='any')
# print(datas.head())
# print(datas.shape)

## 抽取x,y
X = datas[names[:-1]]
Y = datas[names[-1]]

## 分割训练集和测试集

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initiate logistic regression object
logit = linear_model.LogisticRegression()

# Fit model. Let X_train = matrix of predictors, y_train = matrix of variable.
# NOTE: Do not include a column for the intercept when fitting the model.
resLogit = logit.fit(x_train, y_train)

# Calculate matrix of predicted class probabilities.
# Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
predProbs = resLogit.predict_proba(x_train)
gc.collect()
# Design matrix -- add column of 1's at the beginning of your X_train matrix
X_design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
# Initiate matrix of 0's, fill diagonal with each predicted observation's variance
V = np.prod(predProbs, axis=1).ravel()
# V = np.diagflat(np.prod(predProbs, axis=1))


# Covariance matrix
# Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
# Python 3.5+, you can replace the covLogit-line below with the more readable:
# covLogit = np.linalg.inv(X_design.T @ V @ X_design)
covLogit = np.linalg.inv(np.dot(X_design.T * V, X_design))
print("Covariance matrix: ", covLogit)

# Standard errors
print("Standard errors: ", np.sqrt(np.diag(covLogit)))

# Wald statistic (coefficient / s.e.) ^ 2
# logitParams = np.insert(resLogit.coef_, 0, resLogit.intercept_)
# print("Wald statistics: ", (logitParams / np.sqrt(np.diag(covLogit))) ** 2)
