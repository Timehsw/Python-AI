# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/26
    Desc : 单个状态的概率.计算gamma值,即给定模型lambda和观测序列Q的时候,时刻t对应状态i的概率值
'''

import numpy as np

from MachineLearning.HMM.algorithm_implementation import common
from MachineLearning.HMM.algorithm_implementation import forward_probability as forward
from MachineLearning.HMM.algorithm_implementation import backward_probability as backward


def calc_gamma(alpha, beta, gamma):
    '''
    根据alpha和beta的值计算单个状态的概率gamma值
    :param alpha: 前向概率值
    :param beta: 后向概率值
    :param gamma: 结果存储到gamma中
    :return:
    '''
    T = len(alpha)
    n_range = range(alpha.shape[1])
    tmp = np.zeros(T)
    for t in range(T):
        for i in n_range:
            tmp[i] = alpha[t][i] * beta[t][i]
        sum_alpha_beta_of_t = np.sum(tmp)

        # 更新gamma值
        for i in n_range:
            gamma[t][i] = tmp[i] / sum_alpha_beta_of_t


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
    alpha = np.zeros((len(Q), len(A)))
    beta = np.zeros((len(Q), len(A)))
    gamma = np.zeros((len(Q), len(A)))

    # 开始计算
    # 1. 计算后向概率beta矩阵
    backward.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    print("后向概率beta矩阵:\n", beta)
    tmp = 0
    for i in range(len(A)):
        tmp += pi[i] * B[i][common.convert_obs_seq_2_index(Q, 0)] * beta[0][i]
    print(Q, "出现的概率,后向算法--->", tmp)

    # 2. 计算前向概率alpha矩阵
    forward.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    print("前向概率alpha矩阵:\n", alpha)
    print(Q, "出现的概率,前向算法-->", np.sum(alpha[-1]))

    # 3.计算gamma矩阵
    calc_gamma(alpha, beta, gamma)
    print("单个状态概率gamma矩阵:\n", gamma)

    # 选择每个时刻最大的概率作为预测概率
    print('各个时刻最大概率的盒子为:', end='')
    index = ['盒子1', '盒子2', '盒子3']
    for p in gamma:
        print(index[p.tolist().index(np.max(p))], end="\t")
