# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/5/28
    Desc : 基于梯度下降法实现线性回归
'''

# 数据检验
def validate(X,Y):
    if len(X)!=len(Y):
        raise Exception("参数异常")
    else:
        m=len(X[0])
        for l in X:
            if len(l)!=m:
                raise Exception("参数异常")

        if len(Y[0])!=l:
            raise Exception("参数异常")



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
