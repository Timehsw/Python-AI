# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/23
    Desc : 计算dvi
    Note :
'''

import numpy as np


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(k_list):
    '''
    k_list为ndarray,每一个元素是该聚类的所有样本点
    :param k_list:
    :return:
    '''
    """ Dunn index [CVI]

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di
