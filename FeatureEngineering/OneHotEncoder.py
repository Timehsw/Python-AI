# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 哑编码
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

print('-' * 40, 'sklearn哑编码处理')

# OneHotEncoder要求数据类别必须是数值的
encoder = OneHotEncoder()
data = np.array([
    [0, 0, 3],
    [1, 1, 0],
    [0, 2, 1],
    [1, 0, 2],
    [1, 1, 1]])
encoder.fit(data)
print(data)
print('编码结果')

'''
n_values_代表每一个特征上有几个值
data是一个二维数组,每一列代表一个特征,一共三列,那么就有三个特征
第一列为:0,1,0,1,1 一共0,1共2个值
第二列为:0,1,2,0,1 一共有0,1,2共3个值
...类同
'''
print(encoder.n_values_)  # 输出为[2 3 4]
print(encoder.transform([[0, 1, 2]]).toarray())
# 输出为[[1. 0.(第一位2个值,2位表示)| 0. 1. 0. (第二位3个值,3位表示)|0. 0. 1. 0.]]
print('~' * 100)

# sparse:最终产生的结果是否是稀疏化矩阵,默认为True,一般不改动
dv = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2.2}, {'foo': 3, 'baz': 2}]
X = dv.fit_transform(D)
print(D)
print(X)

# 直接把字典中的Key作为特征,value作为特征值,然后构建特征矩阵
print(dv.get_feature_names())
print(dv.transform({'foo': 4, 'unseen': 3}))

print('~' * 100)
hasher = FeatureHasher(n_features=3)

D = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
# 直接以hash值计算结果 -- 该方式一般不用
f = hasher.transform(D)
print(f.toarray())

print('-' * 40, 'pandas哑编码处理')

path = "datas/car.data"
data = pd.read_csv(path, header=None)

for i in range(7):
    print(i, np.unique(data[i]))

### 字符串转换为序列id（数字）
X = data.apply(lambda x: pd.Categorical(x).codes)
print(X.head(5))

### 进行哑编码操作
enc = OneHotEncoder()
X = enc.fit_transform(X)
print(enc.n_values_)

### 转换后数据
df2 = pd.DataFrame(X.toarray())
df2.head(5)

print(df2.info())