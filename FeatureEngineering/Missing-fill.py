# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 缺省值填充
    Link: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer
'''

import numpy as np
from sklearn.preprocessing import Imputer

X = [
    [2, 2, 4, 1],
    [np.nan, 3, 4, 4],
    [1, 1, 1, np.nan],
    [2, 2, np.nan, 3]
]
X2 = [
    [2, 6, np.nan, 1],
    [np.nan, 5, np.nan, 1],
    [4, 1, np.nan, 5],
    [np.nan, np.nan, np.nan, 1]
]

# 按照列进行填充值的计算 axis=0
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)

# 按照行计算填充值(如果按照行进行填充的话,那么是不需要进行模型fit的,直接使用x的现有的行信息进行填充).一般不用
imp2=Imputer(missing_values='NaN',strategy='mean',axis=1)

imp1.fit(X)
imp2.fit(X)

'''
imp1是用列填充,代表的意思是用X样本中的每一列的均值来填充X2中的每一列的缺失值
X第一列有3个值不为空:2+1+2/3=1.66666667
X第二列有4个值:2+3+1+2/4=2
....

这就是按列以及根据均值来填充缺失值的计算方法

imp2是按行填充,都不会对X进行fit求出缺省值.而是直接按行对X2中的每一行求均值进行填充的
'''
print(imp1.statistics_)
# 输出为[1.66666667 2.         3.         2.66666667]
print(np.array(X2))
print('~'*100)
print(imp1.transform(X2))
print('~'*100)
print(imp2.transform(X2))

print('-'*100)
# 分别用均值,中位数,众数来填值
imp3=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp4=Imputer(missing_values='NaN',strategy='median',axis=0)
imp5=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)

imp3.fit(X)
imp4.fit(X)
imp5.fit(X)

print(np.array(X))
print('~'*100)
print(np.array(X2))
print('~'*100)
print(imp3.transform(X2))
print('~'*100)
print(imp4.transform(X2))
print('~'*100)
print(imp5.transform(X2))
