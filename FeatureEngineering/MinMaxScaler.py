# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 区间缩放法/归一化
    说明 : 归一化对于不同特征维度的伸缩变换的主要目的是为了使得维度度量之间特征具有可比性,同时不改变原始数据的分布(相同特性的特征转换后,还是具有相同特性).和标准化一样,也属于一种无量纲化的操作方式
'''

'''
区间缩放法:是指按照数据的取值范围特性对数据进行缩放操作,将数据缩放到给定区间上
转换后的值=(X-X.min)/(X.max-X.min)
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([
    [1, -1, 2, 3],
    [2, 0, 0, 3],
    [0, 1, -10, 3]
], dtype=np.float64)

# feature_range表示缩放范围.feature_range=(0, 1) 表示0-1范围,那么此时也就是归一化了
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
# 每一列代表一个特征
print("每一列的最大值:\n", scaler.data_max_)
print("每一列的最小值:\n", scaler.data_min_)
print("每一列的最大值-最小值:\n", scaler.data_range_)

print("进行区间缩放")
print(scaler.transform(X))
print("~" * 100)
import pandas as pd

print(pd.DataFrame(X).describe())
print("进去区间缩放之后的desc")
print(pd.DataFrame(scaler.transform(X)).describe())
