# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/4
    Desc : sklearn中的metrics用法
'''

import numpy as np
from sklearn.metrics import accuracy_score, recall_score,confusion_matrix

print('-' * 20, ' accuracy_score ', '-' * 20)
# accuracy_score(准确率)
# 分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
score = accuracy_score(y_true, y_pred)
print(score)

# normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
score_num = accuracy_score(y_true, y_pred, normalize=False)
print(score_num)

a = np.array([[0, 1], [1, 1]])
b = np.ones((2, 2))
print(a)
print(b)
print(accuracy_score(a, b))

print('-' * 20, ' recall_score ', '-' * 20)
# 召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了。

# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
#
# recall_score1 = recall_score(y_true, y_pred, average='macro')
# recall_score2 = recall_score(y_true, y_pred, average='micro')
# recall_score3 = recall_score(y_true, y_pred, average='weighted')
# recall_score4 = recall_score(y_true, y_pred, average=None)
# print(recall_score1)
# print(recall_score2)
# print(recall_score3)
# print(recall_score4)
print('-' * 20, ' confusion_matrix ', '-' * 20)

matrix = confusion_matrix(y_true, y_pred)
print(y_true)
print(y_pred)

print(matrix)
