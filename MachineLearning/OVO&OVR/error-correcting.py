# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/18
    Desc : Error-Correcting案例代码
'''

from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 数据获取
iris = datasets.load_iris()
x, y = iris.data, iris.target
print('样本数量:%d,特征数量:%d' % x.shape)

# 模型对象构建
# code_size
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=30, random_state=0)
clf.fit(x, y)

# 输出预测结果值
print(clf.predict(x))
print('准确率:%.3f' % accuracy_score(y, clf.predict(x)))

# 模型属性输出
k = 1
for item in clf.estimators_:
    print('第%d个模型:' % k)
    print(item)
    k += 1
print(clf.classes_)
