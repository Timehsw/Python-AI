# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/25
    Desc : 前向算法
'''

import numpy as np

from MachineLearning.HMM.algorithm_implementation import common


def calc_alpha(pi, A, B, Q, alpha, fetch_index_by_obs_seq=None):
    '''
    NOTE:ord函数的含义是将单个的字符转换为相应的ASCII码
    计算前向概率的alpha值
    :param pi: 初始的随机概率值
    :param A: 状态转移矩阵
    :param B: 状态和观测值之间的转移矩阵
    :param Q: 观测值列表
    :param alpha: 前向概率的alpha矩阵
    :param fetch_index_by_obs_seq: 根据序列获取对应的索引值,可以为空
    :return:
    '''
    # 0.初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 1. 初始化一个状态类别的顺序
    n = len(A)
    n_range = range(n)

    # 2. 更新初值(t=1)
    for i in n_range:
        alpha[0][i] = pi[i] * B[i][fetch_index_by_obs_seq_f(Q, 0)]

    # 3. 迭代更新其它时刻
    T = len(Q)
    tmp = [0 for i in n_range]
    for t in range(1, T):
        for i in n_range:
            # 1. 计算上一个时刻t-1累积过来的概率值
            for j in n_range:
                tmp[j] = alpha[t - 1][j] * A[j][i]

            # 2. 更新alpha的值
            alpha[t][i] = np.sum(tmp) * B[i][fetch_index_by_obs_seq_f(Q, t)]

    return alpha


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    Q = '白黑白白黑'
    alpha = np.zeros((len(Q), len(A)))
    # 开始计算
    calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)

    print(alpha)
    res = np.sum(alpha[-1])
    print(Q, " --> ", res)
