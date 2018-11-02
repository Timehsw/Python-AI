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
predProbs = np.matrix(resLogit.predict_proba(x_train))

# Design matrix -- add column of 1's at the beginning of your X_train matrix
# X_design = numpy.hstack((numpy.ones(shape = (x_train.shape[0],1)), X))
X_design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
# Initiate Matrix of 0's, fill diagonal with each predicted observation's variance
V = np.matrix(np.zeros(shape=(X_design.shape[0], X_design.shape[0])))
np.fill_diagonal(V, np.multiply(predProbs[:, 0], predProbs[:, 1]).A1)
V = np.prod(predProbs, axis=1).ravel()

# Covariance matrix
covLogit = np.linalg.inv(X_design.T * V * X_design)
print('Covariance matrix: ', covLogit)

# Standard errors
print('Standard errors: ', np.sqrt(np.diag(covLogit)))

# Wald statistic (coefficient / s.e.) ^ 2
# logitParams = numpy.insert(resLogit.coef_, 0, resLogit.intercept_)
# print('Wald statistics: ', (logitParams / numpy.sqrt(numpy.diag(covLogit))) ** 2)
