# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/12
    Desc : sigmoid函数
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./datas/iris.data', header=None)
df.columns = ['a', 'b', 'c', 'd', 'class']
print(df.head())
print('查看有几个类别:\n', df['class'].value_counts())
df['class'] = pd.Categorical(df['class']).codes
print('对类别字符串进行转码:\n', df['class'].value_counts())
df = df[df['class'] != 2]
print('排除掉第2类:\n', df['class'].value_counts())
x = df.drop(['class'], 1)
y = df['class']
print('x的特征:\n', x.head())
print('y的类别:\n', y.head())


# 使用梯度下降法求解

# 概率转换成01的类数据
def prob2class(y_prob):
    y_class = [1 if i >= 0.5  else 0 for i in y_prob]
    return y_class


# sigmoid函数
def sigmoid(theta, x):
    prob = 1 / (1 + np.exp(-x.dot(theta)))
    return prob.values


# 损失函数
def lr_loss(y_true, prob):
    laplace = 1e-10
    lr_loss = - sum(y_true * np.log(prob).ravel() + (1 - y_true) * np.log(1 - prob + laplace).ravel()) + 1 / len(
        y) * np.power(theta, 2).sum()
    return lr_loss


# 初始化aplha 和 theta
alpha = 0.02
theta = np.zeros((4, 1))
theta

# 梯度下降法开始迭代
y_true = y.reshape(-1, 1)
lr_prob = sigmoid(theta, x)
# print(lr_prob)
lr_class = prob2class(lr_prob)
print(lr_class)
theta = theta + alpha * x.T.dot(y_true - sigmoid(theta, x))
lr_loss(y, lr_prob)

x = np.random.randint(1, 5, (5, 1))
y = np.random.randint(1, 5, (5, 1))
print(x)
print(y)


def sigmoid_func1(theta, x):
    return 1 / (1 + np.exp(-theta.T.dot(x)))


def sigmoid_func(z):
    return 1 / (1 + np.exp(-z))


x = pd.Series(np.arange(-10, 10, 0.01))
y = x.apply(lambda x: sigmoid_func(x))

plt.figure(facecolor='w')
plt.plot(x, y, 'r-', linewidth=2, label="content")
plt.legend(loc='lower right')
plt.title("sigmoid 函数图像")
plt.grid(b=True)
plt.show()
