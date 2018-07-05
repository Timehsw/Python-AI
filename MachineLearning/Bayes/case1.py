# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/18
    Desc : 贝叶斯算法案例一/鸢尾花数据分类
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB, MultinomialNB  # 高斯贝叶斯和多项式朴素贝叶斯
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

iris_feature = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
features = [2, 3]
path = 'datas/iris.data'
data = pd.read_csv(path, header=None)
x = data[features]
y = pd.Categorical(data[4]).codes
print('总样本数:%d ; 特征属性数目:%d' % x.shape)

# 数据划分
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1

# 高斯贝叶斯模型构建
clf = Pipeline([
    ('sc', StandardScaler()),  # 标准化,把它转化成高斯分布
    ('poly', PolynomialFeatures(degree=4)),
    ('clf', GaussianNB())  # MultinomiaNB多项式贝叶斯算法中要求特征属性的取值不能为负数
])

# 训练模型
clf.fit(x_train, y_train)

# 计算预测值并计算准确率
y_train_hat = clf.predict(x_train)
print('训练集准确度:%.2f%%' % (100 * accuracy_score(y_train, y_train_hat)))
y_test_hat = clf.predict(x_test)
print("测试集准确度:%.2f%%" % (100 * accuracy_score(y_test, y_test_hat)))

# 产生区域图
# 纵横采样多少个值
N, M = 500, 500
x1_min1, x2_min1 = x_train.min()
x1_max1, x2_max1 = x_train.max()
x1_min2, x2_min2 = x_test.min()
x1_max2, x2_max2 = x_test.max()
x1_min = np.min((x1_min1, x1_min2))
x1_max = np.max((x1_max1, x1_max2))
x2_min = np.min((x2_min1, x2_min2))
x2_max = np.max((x2_max1, x2_max2))

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
# 生成网格采样点
x1, x2 = np.meshgrid(t1, t2)
# 测试点
x_show = np.dstack((x1.flat, x2.flat))[0]

cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_show_hat = clf.predict(x_show)
y_show_hat = y_show_hat.reshape(x1.shape)

# 画图
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x_train[features[0]], x_train[features[1]], c=y_train, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[features[0]], x_test[features[1]], c=y_test, marker='^', edgecolors='k', s=120, cmap=cm_dark)

plt.xlabel(iris_feature[features[0]], fontsize=13)
plt.ylabel(iris_feature[features[1]], fontsize=13)

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('GaussianNB对鸢尾花的分类结果,正确率:%.3f%%' % (100 * accuracy_score(y_test, y_test_hat)), fontsize=12)
plt.grid(True)
plt.show()
