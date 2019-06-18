# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/18
    Desc : OVR案例代码
'''

import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score

# 加载数据
iris = datasets.load_iris()
x, y = iris.data, iris.target
print('样本数量:%d, 特征数量:%d' % x.shape)
print("label分类个数: ", np.unique(y))

# ovr模型构建
clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(x, y)

# 预测结果输出
print(clf.predict(x))
print('准确率:%.3f' % accuracy_score(y, clf.predict(x)))
print('准确率:%.3f' % precision_score(y, clf.predict(x)))

# 模型属性输出
k = 1
for item in clf.estimators_:
    print('第%d个模型' % k)
    print(item)
    k += 1
print(clf.classes_)
