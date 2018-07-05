# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 多项式转换
'''

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
print(X)

'''
当degree=2时候,
假设输入特征属性为[a,b]
那么输出的多项式特征为[1,a,b,a^2,ab,b^2]
'''
poly1 = PolynomialFeatures(degree=2)
poly1.fit(X)
print(poly1)
print(poly1.transform(X))

print('-' * 100)

# 当interaction_only设置为true的时候,表示不使用单个变量的多次项扩充维度
# 也就是去掉a^2,b^2这种高次项,只要两两组合的
poly2 = PolynomialFeatures(interaction_only=True)
poly2.fit(X)
print(poly2)
print(poly2.transform(X))

print('-' * 100)

# 当include_bias设置为False时候,表示不加入常数项1
poly3 = PolynomialFeatures(include_bias=False)
poly3.fit(X)
print(poly3)
print(poly3.transform(X))
