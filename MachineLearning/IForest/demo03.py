# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/15
    Desc : 孤立森林例子
    Note : 多维数据,超过2维了
'''

import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# 生成训练数据
X = 0.3 * rng.randn(100, 3)
X_train = np.r_[X + 1, X - 3, X - 5, X + 6]

# 生成正常数据
X = 0.3 * rng.randn(20, 3)
X_test = np.r_[X + 1, X - 3, X - 5, X + 6]

# 生成异常数据
X_outliers = rng.uniform(low=-8, high=8, size=(20, 3))

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

# 作图
# xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
# Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
# Z=Z.reshape(xx.shape)


plt.title('IsolationForest')
# plt.contourf(xx,yy,Z,cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
plt.axis('tight')
plt.xlim((-8, 8))
plt.ylim((-8, 8))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
