# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 正则化
    说明 : 正则化则是通过范数规则来约束特征属性,通过正则化我们可以降低数据训练出来的模型的过拟合可能. 在进行正则化操作的过程中,不会改变数据的分布情况,但是会改变数据特征之间的相关特性.
'''

'''
正则化:和标准化不同,正则化是基于矩阵的行进行数据处理,其目的是将矩阵的行均转换为"单位向量"
'''
import numpy as np
from sklearn.preprocessing import Normalizer

X = np.array([
    [1, -1, 2],
    [2, 0, 0],
    [0, 1, -10]
], dtype=np.float64)

normalizer1 = Normalizer(norm='max')
normalizer2 = Normalizer(norm='l2')

normalizer1.fit(X)
normalizer2.fit(X)

# 转换
print(normalizer1.transform(X))
print('~' * 100)
print(normalizer2.transform(X))

print('-' * 100)
import pandas as pd

print(pd.DataFrame(X).describe())
print(pd.DataFrame(normalizer2.transform(X)).describe())
