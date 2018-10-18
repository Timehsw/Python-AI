# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/5/28
    Desc : Logistic案例:乳腺癌分类
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

## 数据读取并处理异常数据

path = './datas/breast-cancer-wisconsin.data'
names = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']


# path = './datas/demo.csv'
# names = ['id','checking','duration','history','purpose','amount','savings','employed','installp','marital','coapp','resident','property','age','other','housing','existcr','job','depends','telephon','foreign','good_bad','checkingstr','historystr']
df = pd.read_csv(path, names=names)
print(df.head())
print(df.shape)
datas = df.replace('?', np.nan).dropna(how='any')
# print(datas.head())
# print(datas.shape)

## 抽取x,y
Y = datas['Class']
X=datas.drop(columns=['Class'])

## 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

## 数据归一化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
print(X_train[0:3])

## 模型构建
'''
penalty:过拟合解决参数,l1或者l2
solver:参数优化方式

'''
lr = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2',
                          solver='lbfgs', tol=0.01)
re = lr.fit(X_train, Y_train)
# lr.fit(X_train,Y_train)
# 4. 模型效果获取
r = re.score(X_train, Y_train)
print("R值（准确率）：", r)
print("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("参数：", re.coef_)
print("截距：", re.intercept_)
# print(re.predict_proba(X_test))  # 获取sigmoid函数返回的概率值

# 数据预测
## a. 预测数据格式化(归一化)
X_test = ss.transform(X_test)  # 使用模型进行归一化操作
## b. 结果数据预测
Y_predict = re.predict(X_test)
Y_predict2 = lr.predict_proba(X_test)
# print(re.score(X_test, Y_test))

from BasicLibs.learnMatplotlib.binary_evaluation import binary_evaluation

utils=binary_evaluation(Y_test.values,Y_predict2[:, 1])
utils.plot_ks()