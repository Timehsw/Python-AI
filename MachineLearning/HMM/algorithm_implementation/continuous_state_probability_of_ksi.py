# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/26
    Desc : 两个状态的联合概率.计算ksi值,即给定模型lambda和观测序列Q的时候,时刻t对应状态i并时刻t+1处于状态j的概率值
'''

import numpy as np

from MachineLearning.HMM.algorithm_implementation import common
from MachineLearning.HMM.algorithm_implementation import forward_probability as forward
from MachineLearning.HMM.algorithm_implementation import single_state_probability_of_gamma as single
from MachineLearning.HMM.algorithm_implementation import backward_probability as backward


def calc_ksi(alpha, beta, A, B, Q, ksi, fetch_index_by_obs_seq=None):
    '''
    计算时刻t的时候状态为i,时刻t+1的时候状态位j的联合概率ksi
    :param alpha: 对应的前向概率值
    :param beta: 对应的后向概率值
    :param A: 状态转移矩阵
    :param B: 状态和观测值之间的转移矩阵
    :param Q: 观测值列表
    :param ksi: 待求解的ksi矩阵
    :param fetch_index_by_obs_seq:
    :return:
    '''
    # 0.初始化
    fetch_index_by_obs_seq_f = fetch_index_by_obs_seq
    if fetch_index_by_obs_seq_f is None:
        fetch_index_by_obs_seq_f = lambda obs, obs_index: ord(obs[obs_index])

    T = len(Q)
    n = len(A)

    # 1.开始迭代更新
    n_range = range(n)
    tmp = np.zeros((n, n))

    for t in range(T - 1):
        # 1.计算t时刻状态为i,t+1时刻状态为j的概率值
        for i in n_range:
            for j in n_range:
                tmp[i][j] = alpha[t][i] * A[i][j] * B[j][fetch_index_by_obs_seq_f(Q, t + 1)] * beta[t + 1][j]

        # 2. 计算t时刻的联合概率和
        sum_pro_of_t = np.sum(tmp)

        # 3. 计算时刻t的联合概率ksi
        for i in n_range:
            for j in n_range:
                ksi[t][i][j] = tmp[i][j] / sum_pro_of_t


if __name__ == '__main__':
    # 初始矩阵
    pi = np.array([0.2, 0.5, 0.3])
    # 状态转移矩阵
    A = np.array([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]])
    # 发射矩阵
    B = np.array([[0.4, 0.6], [0.8, 0.2], [0.5, 0.5]])
    # 观测序列
    Q = "白黑白白黑"

    # TxN矩阵
    T = len(Q)
    n = len(A)
    beta = np.zeros((T, n))
    alpha = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))

    # 开始计算
    # 1. 计算后向概率beta矩阵
    backward.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    print("后向概率beta矩阵:\n", beta)
    tmp = 0
    for i in range(len(A)):
        tmp += pi[i] * B[i][common.convert_obs_seq_2_index(Q, 0)] * beta[0][i]
    print(Q, "出现的概率,后向算法--->", tmp)

    print('~' * 100)

    # 2. 计算前向概率alpha矩阵
    forward.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    print("前向概率alpha矩阵:\n", alpha)
    print(Q, "出现的概率,前向算法-->", np.sum(alpha[-1]))

    print('~' * 100)

    # 3.计算gamma矩阵
    single.calc_gamma(alpha, beta, gamma)
    print("单个状态概率gamma矩阵:\n", gamma)

    # 选择每个时刻最大的概率作为预测概率
    print('各个时刻最大概率的盒子为:', end='')
    index = ['盒子1', '盒子2', '盒子3']
    for p in gamma:
        print(index[p.tolist().index(np.max(p))], end="\t")

    print()
    print('~' * 100)
    # 4. 计算ksi矩阵
    calc_ksi(alpha, beta, A, B, Q, ksi, common.convert_obs_seq_2_index)
    print('两个状态的联合概率ksi矩阵:\n', ksi)
