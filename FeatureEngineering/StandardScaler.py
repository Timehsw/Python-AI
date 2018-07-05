# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 标准化
    说明 : 标准化的目的是为了降低不同特征的不同范围的取值对于模型训练的影响
    好处 : 1.提高迭代求解的收敛速度;2.提高迭代求解的精度.
'''

'''
标准化:基于特征属性的数据(也就是特征矩阵的列),获取均值和方差,然后将特征值转换至服从标准正态分布.
样本值减去均值除以方差
'''

from sklearn.preprocessing import StandardScaler

X = [
    [1, 2, 3, 2],
    [7, 8, 9, 2.01],
    [4, 8, 2, 2.01],
    [9, 5, 2, 1.99],
    [7, 5, 3, 1.99],
    [1, 4, 9, 2]
]

ss = StandardScaler(with_mean=True, with_std=True)
ss.fit(X)
print(ss.mean_)
print(ss.n_samples_seen_)
print(ss.scale_)
print(ss.transform(X))
print('~' * 100)

import pandas as pd

df = pd.DataFrame(X)
print(df.describe())

print('~' * 100)
df2 = pd.DataFrame(ss.transform(X))
print(df2.describe())
