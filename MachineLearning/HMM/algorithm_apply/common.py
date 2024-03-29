# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/26
    Desc : 公共类
'''
import math
import numpy as np

infinite = float(-2 ** 31)


def log_sum_exp(a):
    '''
    参考numpy中的log sum exp 的api
    scipy.misc.logsumexp
    :param a:
    :return:
    '''
    a = np.asarray(a)
    a_max = max(a)
    tmp = 0
    for k in a:
        tmp += math.exp(k - a_max)
    return a_max + math.log(tmp)


def convert_obs_seq_2_index(Q, index=None):
    """
    将观测序列转换为观测值的索引值
    Q:是输入的观测序列
    """
    if index is not None:
        cht = Q[index]
        if "黑" == cht:
            return 1
        else:
            return 0
    else:
        result = []
        for q in Q:
            if "黑" == q:
                result.append(1)
            else:
                result.append(0)
        return result

if __name__ == '__main__':
    arr=[1,2,3]
    exp = log_sum_exp(arr)
    print(exp)
