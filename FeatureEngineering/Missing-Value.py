# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc :
'''

import pandas as pd

df = pd.read_csv('./datas/missvaluesampledata.csv', header=None)
print(df)

print('-' * 20, "查看空值的分布情况", '-' * 20)

print('-' * 20, "按列统计空值数", '-' * 20)
print(df.isnull().sum())

print('-' * 20, "按行统计空值数", '-' * 20)
print(df.isnull().sum(axis=1))

print('-' * 20, "对空值进行处理,删除空值", '-' * 20)
# 处理缺失值最简单的方法就是，将包含缺失值数据的列或者行从数据中删除，但这样会造成数据的浪费。

print('-' * 20, "删除包含缺失值的行", '-' * 20)
print(df.dropna())

print('-' * 20, "删除包含缺失值的列", '-' * 20)
print(df.dropna(axis=1))

# 在使用dropna方法的时候，我们可以通过设置inplace=True直接修改data的值，默认是是Flase。
print(df)