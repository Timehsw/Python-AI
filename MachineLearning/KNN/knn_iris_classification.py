# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/2
    Desc : 用KNN做鸢尾花数据分类
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

path = './datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df = pd.read_csv(path, header=None, names=names)
print(df['class'].value_counts())
print(df.head())

print('-' * 50)


# df['label'] = pd.CategoricalIndex(df['class']).codes
# print(df['label'].value_counts())
# print(df.head())

# 自定义
def parseRecord(record):
    result = []
    r = zip(names, record)
    for name, v in r:
        if name == 'class':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result

# 数据转换为数字以及分割
# 数据转换
datas = df.apply(lambda r: parseRecord(r), axis=1)
datas = datas.dropna(how='any')
# 数据分割,获取x,y
X=datas[names[:-1]]
Y=datas[names[-1]]

# 训练集测试集划分
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("训练集数据为:" , X_train.shape)
print("测试集数据为:" , X_test.shape)

# 模型构建
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

# b. 模型效果输出
## 将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test,classes=(1,2,3))
## 得到预测属于某个类别的概率值
knn_y_score = knn.predict_proba(X_test)
## 计算roc的值
knn_fpr, knn_tpr, knn_threasholds = metrics.roc_curve(y_test_hot.ravel(),knn_y_score.ravel())
## 计算auc的值
knn_auc = metrics.auc(knn_fpr, knn_tpr)
print ("KNN算法训练集R值：", knn.score(X_train, Y_train))
print ("KNN算法AUC值：", knn_auc)

# 模型预测
knn_y_predict = knn.predict(X_test)
print('KNN算法测试集R值：',knn.score(X_test,Y_test))
print("KNN",metrics.accuracy_score(Y_test,knn_y_predict))
## 画图2：预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, Y_test, 'ro',markersize = 6, zorder=3, label=u'真实值')
plt.plot(x_test_len, knn_y_predict, 'yo', markersize = 16, zorder=1, label=u'KNN算法预测值,$R^2$=%.3f' % knn.score(X_test, Y_test))
plt.legend(loc = 'lower right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.show()