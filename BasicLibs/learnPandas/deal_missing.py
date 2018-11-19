# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/16
    Desc : 
    Note : 
'''

import numpy as np
import pandas as pd

# path = '/Users/hushiwei/GEO/数据集/建模数据集/Newmodeling.csv'
path = '/Users/hushiwei/GEO/数据集/建模数据集/cleand_1416c.csv'

df = pd.read_csv(path)
# print(df.isnull().any())

# 统计每一列缺失值的个数
missing_value_count_by_columns = df.shape[0] - df.count()
# 统计每一列的空值率
missing_value_count_by_columns = missing_value_count_by_columns.map(lambda x: x / df.shape[0])
print(df.shape)
print(missing_value_count_by_columns)
# print(type(missing_value_count_by_columns))

print('~' * 50, '空值率在70%一下的列', '~' * 50)
'''
2089列全为空
缺失率在70%一下的还有1417列
'''
usefule_missing = missing_value_count_by_columns[missing_value_count_by_columns <0.7]
print(usefule_missing)

# usefule_columns=usefule_missing.index.values.tolist()
# usefule_columns.remove('Unnamed: 0')
# print(usefule_columns)
# mean_value=df[usefule_columns].mean()
# print(mean_value)
# df=df[usefule_columns].fillna(mean_value)
# df.to_csv('/Users/hushiwei/GEO/数据集/建模数据集/cleand_1416c.csv',index=False)

# 用方差大小来做特征选择
from sklearn.feature_selection import VarianceThreshold

VarianceThreshold()