# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/6
    Desc : Bagging&Boosting 算法应用在回归模型中
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def notEmpty(s):
    return s != ''


## 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "./datas/boston_housing.data"
## 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再安装每行数据进行处理
fd = pd.read_csv(path, header=None)
# print (fd.shape)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):  # enumerate生成一列索 引i,d为其元素

    d = map(float, filter(notEmpty, d[0].split(' ')))  # filter一个函数，一个list

    # 根据函数结果是否为真，来过滤list中的项。
    data[i] = list(d)

## 分割数据
x, y = np.split(data, (13,), axis=1)
print(x[0:5])
y = y.ravel()  # 转换格式 拉直操作
print(y[0:5])
ly = len(y)
print(y.shape)
print("样本数据量:%d, 特征个数：%d" % x.shape)
print("target样本数据量:%d" % y.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

print('-' * 20, '线性回归模型', '-' * 20)

## 线性回归模型
linear = LinearRegression()
linear.fit(x_train, y_train)
print("训练集上R^2:%.5f" % linear.score(x_train, y_train))
print("测试集上R^2:%.5f" % linear.score(x_test, y_test))

print('-' * 20, 'Ridge回归模型', '-' * 20)

## Ridge回归模型
lr = Ridge(alpha=0.1)
lr.fit(x_train, y_train)
print("训练集上R^2:%.5f" % lr.score(x_train, y_train))
print("测试集上R^2:%.5f" % lr.score(x_test, y_test))

print('-' * 20, '使用Bagging思想集成线性回归', '-' * 20)

# 使用Bagging思想集成线性回归
bg = BaggingRegressor(Ridge(alpha=0.1), n_estimators=50, max_samples=0.7, max_features=0.8, random_state=28)
bg.fit(x_train, y_train)
print("训练集上R^2:%.5f" % bg.score(x_train, y_train))
print("测试集上R^2:%.5f" % bg.score(x_test, y_test))

print('-' * 20, '使用AdaBoostRegressor', '-' * 20)

# 使用AdaBoostRegressor
adr = AdaBoostRegressor(LinearRegression(), n_estimators=100, learning_rate=0.001, random_state=14)
adr.fit(x_train, y_train)
print("训练集上R^2:%.5f" % adr.score(x_train, y_train))
print("测试集上R^2:%.5f" % adr.score(x_test, y_test))

print('-' * 20, 'GradientBoostingRegressor', '-' * 20)

# 使用AdaBoostRegressor; GBDT模型只支持CART模型
gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=14)
gbdt.fit(x_train, y_train)
print("训练集上R^2:%.5f" % gbdt.score(x_train, y_train))
print("测试集上R^2:%.5f" % gbdt.score(x_test, y_test))
