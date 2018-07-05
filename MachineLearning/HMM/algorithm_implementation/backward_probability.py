# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/25
    Desc : 后向概率,后向算法
    后向算法,反着路径算,结合代码再好好看看
'''
import numpy as np

from MachineLearning.HMM.algorithm_implementation import common


def calc_beta(pi, A, B, Q, beta, fetch_index_by_obs_seq=None):
    '''
    计算后向概率beta的值
    :param pi: 初始的随机概率值
    :param A: 状态转移概率矩阵
    :param B: 状态和观测值之间的转移矩阵
    :param Q: 观测值列表
    :param beta: 后向概率beta矩阵
    :param fetch_index_by_obs_seq:
    :return:
    '''
    # 0.初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    # 1. 初始化一个状态类别的顺序
    n = len(A)
    n_range = range(n)
    T = len(Q)

    # 2. 更新初值t=T
    for i in n_range:
        beta[T - 1][i] = 1

    # 3. 迭代更新其它时刻
    tmp = np.zeros(n)
    for t in range(T - 2, -1, -1):
        for i in n_range:
            for j in n_range:
                tmp[j] = A[i][j] * beta[t + 1][j] * B[j][fetch_index_by_obs_seq_f(Q, t + 1)]
            beta[t][i] = np.sum(tmp)

    return beta


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]])
    B = np.array([[0.4, 0.6], [0.8, 0.2], [0.5, 0.5]])
    Q = "白黑白白黑"
    beta = np.zeros((len(Q), len(A)))
    calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    print(beta)

    # 计算最后的概率值
    p = 0
    for i in range(len(A)):
        p += pi[i] * B[i][common.convert_obs_seq_2_index(Q, 0)] * beta[0][i]
    print(Q, "--> 出现的概率为: ", p)
