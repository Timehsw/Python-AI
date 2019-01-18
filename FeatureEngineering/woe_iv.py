# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-01-18
    Desc : 计算woe和iv
    Note : 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# 计算WOE
def woe_transfer(df, var):
    grouped = df['label'].groupby(df[var])
    df_agg = grouped.agg(['sum', 'count'])
    t1 = df_agg['sum'].sum()
    t0 = df_agg['count'].sum() - t1
    n_c = len(df_agg['sum'])
    x_woe = range(n_c)
    ans = {}
    df[var + '_woe'] = df[var]
    for i in range(n_c):
        t1_i = float(df_agg.iloc[i, 0])
        t0_i = float(df_agg.iloc[i, 1] - t1_i)
        if t1_i == 0 and t0_i != 0:
            x_woe[i] = -1
        elif t1_i != 0 and t0_i == 0:
            x_woe[i] = 1
        elif t1_i == 0 and t0_i == 0:
            x_woe[i] = 0
        else:
            x_woe[i] = math.log((t1_i / t1) / (t0_i / t0))
        ans[df_agg.index[i]] = x_woe[i]

        df[var + '_woe'].replace(df_agg.index[i], x_woe[i], inplace=True)

    return [df, ans]


# iv 最大二分类
def binary_iv_class(dset, t1, t0, cnt):
    iv_max = 0
    idex = 0
    dset_t1 = float(dset['sum'].sum())
    dset_t0 = float(dset['count'].sum() - dset_t1)
    print(t1, t0, dset_t1, dset_t0)
    iv0 = (dset_t1 / t1 - dset_t0 / t0) * math.log((dset_t1 / t1) / (dset_t0 / t0))
    for i in range(len(dset.index) - 1):
        dset1 = dset[dset.index <= dset.index[i]]
        dset2 = dset[dset.index > dset.index[i]]
        if dset1['count'].sum() > cnt and dset2['count'].sum() > cnt:

            dset1_t1 = float(dset1['sum'].sum())
            dset1_t0 = float(dset1['count'].sum() - dset1_t1)
            dset2_t1 = float(dset2['sum'].sum())
            dset2_t0 = float(dset2['count'].sum() - dset2_t1)
            if dset1_t1 * dset1_t0 == 0:
                iv1 = 1
            else:
                iv1 = (dset1_t1 / t1 - dset1_t0 / t0) * math.log((dset1_t1 / t1) / (dset1_t0 / t0))
            if dset2_t1 * dset2_t0 == 0:
                iv2 = 1
            else:
                iv2 = (dset2_t1 / t1 - dset2_t0 / t0) * math.log((dset2_t1 / t1) / (dset2_t0 / t0))
            iv = iv1 + iv2
            if iv > iv_max:
                iv_max = iv
                idex = dset.index[i]
                if idex == 11215:
                    print(dset1['count'].sum(), dset2['count'].sum())
    iv_inc = iv_max - iv0
    return [idex, iv_max, iv_inc]


# iv最大 变量分箱  箱数：n_group

def var_class(dset, var, n_group, label, cnt):
    dset = dset.sort_values(by=var)
    #    cnt = len(dset)/500
    df = dset.ix[:, [var, label]]
    t1 = df[label].sum()
    t0 = df[label].count() - t1
    gp = df[label].groupby(df[var])
    df = gp.agg(['sum', 'count'])
    df_t1 = float(df['sum'].sum())
    df_t0 = float(df['count'].sum() - df_t1)
    iv = (df_t1 / t1 - df_t0 / t0) * math.log((df_t1 / t1) / (df_t0 / t0))

    if len(df.index) < n_group:
        n_group = len(df.index)

    ite = 2
    bins = [-float('inf'), float('inf')]
    lab = [str(i) for i in range(len(bins) - 1)]
    var_class_dic = {1: (bins, iv)}
    while ite <= n_group:
        df[var] = pd.cut(df.index, bins, labels=lab)
        iv_inc = 0
        for i_group in range(ite - 1):
            df_i = df[df[var] == str(i_group)][['sum', 'count']]
            if len(df_i) > 1 and df_i['count'].sum() > cnt:
                [max_idex, iv_max_i, iv_inc_i] = binary_iv_class(df_i, t1, t0, cnt)
                if iv_inc_i > iv_inc:
                    iv_inc = iv_inc_i
                    iv_node = max_idex

        if iv_inc == 0:
            break
        iv = iv + iv_inc
        bins = bins + [iv_node]
        bins.sort()
        lab = [str(i) for i in range(len(bins) - 1)]
        var_class_dic[ite] = (bins, iv)
        ite += 1
    return var_class_dic


def is_single_peak(woe_diff_flag):
    if abs(sum(woe_diff_flag)) == len(woe_diff_flag) - 2:
        return True
    else:
        return False


def candidate_index(woe_diff_flag):
    index = [i for i in range(len(woe_diff_flag))]
    if len(woe_diff_flag) > 2:
        for i in range(1, len(woe_diff_flag) - 1):
            if woe_diff_flag[i] == woe_diff_flag[i - 1] and woe_diff_flag[i] == woe_diff_flag[i + 1]:
                if i in index:
                    index.remove(i)
    return index


def is_rank(woe_diff_flag):
    if abs(sum(woe_diff_flag)) == len(woe_diff_flag):
        return True
    else:
        return False


def count_woe_iv(df, var, bins, label):
    lab = [i for i in range(len(bins) - 1)]
    df[var + '_lab'] = pd.cut(df[var], bins, lab)
    df_agg = df[label].groupby(df[var + '_lab']).agg(['sum', 'count'])
    t1 = df[label].sum()
    t0 = df[label].count() - t1
    n_c = len(df_agg['count'])
    x_woe = np.zeros(n_c)
    x_iv = np.zeros(n_c)
    for i in range(n_c):
        t1_i = float(df_agg.iloc[i, 0])
        t0_i = float(df_agg.iloc[i, 1] - t1_i)
        if t1_i == 0 and t0_i != 0:
            x_woe[i] = -1
            x_iv[i] = 1
        elif t1_i != 0 and t0_i == 0:
            x_woe[i] = 1
            x_iv[i] = 1
        elif t1_i == 0 and t0_i == 0:
            x_woe[i] = 0
            x_iv[i] = 0
        else:
            x_woe[i] = np.log((t1_i / t1) / (t0_i / t0))
            x_iv[i] = ((t1_i / t1) - (t0_i / t0)) * np.log((t1_i / t1) / (t0_i / t0))
    df_agg['woe'] = x_woe
    df_agg['iv'] = x_iv
    df_agg['woe_shift'] = df_agg['woe'].shift(1)
    df_agg['woe_shift'].fillna(-9, inplace=True)
    df_agg['woe_diff'] = df_agg.apply(lambda x: 0 if x['woe_shift'] == -9 else x['woe'] - x['woe_shift'], axis=1)
    df_agg['woe_diff_flag'] = df_agg['woe_diff'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    iv = df_agg['iv'].sum()
    woe_diff_flag = list(df_agg['woe_diff_flag'])
    woe_diff_flag.remove(0)
    return iv, woe_diff_flag


def var_merge(df, var, bins_old, label):
    iv, woe_diff_flag = count_woe_iv(df, var, bins_old, label)
    flag = is_rank(woe_diff_flag)

    idx_rank_max = -1
    idx_norank_max = -1
    bins = bins_old[:]
    iv_rank_max = iv
    while len(bins) > 2 and not flag:
        iv_rank_max = 0
        iv_norank_max = 0
        iv, woe_diff_flag = count_woe_iv(df, var, bins, label)
        print(woe_diff_flag)
        idx = candidate_index(woe_diff_flag)
        print(1, idx)
        if len(idx) > 0:
            for i in idx:
                bin_i = bins[:]
                bin_i.remove(bins[i + 1])
                iv_i, woe_diff_flag_i = count_woe_iv(df, var, bin_i, label)
                print(2, is_rank(woe_diff_flag_i))
                if is_rank(woe_diff_flag_i):
                    flag = True
                    if iv > iv_rank_max:
                        iv_rank_max = iv
                        idx_rank_max = i + 1
                else:
                    print(3, iv_norank_max)
                    print(4, iv)
                    if iv > iv_norank_max:
                        iv_norank_max = iv
                        idx_norank_max = i + 1
            if flag == True:
                print(5, flag)
                print(6, bins_old[idx_rank_max])
                print(7, bins)
                if bins_old[idx_rank_max] in bins:
                    bins.remove(bins_old[idx_rank_max])
                print(8, bins)
            else:
                print(9, flag)
                print(10, idx_norank_max)
                print(11, bins_old[idx_norank_max])
                print(bins)
                if bins_old[idx_norank_max] in bins:
                    bins.remove(bins_old[idx_norank_max])
                print(12, bins)
            print(13, bins)
            bins_old = bins[:]

    return {var: [bins, iv_rank_max]}


def Cal_woe_iv(df, var, bins1, label):
    lab = [i for i in range(len(bins1) - 1)]
    df[var + '_lab'] = pd.cut(df[var], bins1, lab)
    df_agg = df[label].groupby(df[var + '_lab']).agg(['sum', 'count'])
    n_c = len(df_agg['sum'])
    t1 = df_agg['sum'].sum()
    t0 = df_agg['count'].sum() - t1
    x_woe = np.zeros(n_c)
    x_iv = np.zeros(n_c)

    for i in range(n_c):
        t1_i = float(df_agg.iloc[i, 0])
        t0_i = float(df_agg.iloc[i, 1] - t1_i)
        if t1_i == 0 and t0_i != 0:
            x_woe[i] = -1
        elif t1_i != 0 and t0_i == 0:
            x_woe[i] = 1
        elif t1_i == 0 and t0_i == 0:
            x_woe[i] = 0
        else:
            x_woe[i] = np.log((t1_i / t1) / (t0_i / t0))
            x_iv[i] = ((t1_i / t1) - (t0_i / t0)) * np.log((t1_i / t1) / (t0_i / t0))
    df_agg['woe'] = x_woe
    df_agg = df_agg.reset_index()
    return df_agg


def plot_woe(df_agg, var):
    fig, axe1 = plt.subplots()
    axe2 = axe1.twinx()
    x = np.arange(len(df_agg))
    y1 = df_agg['count'].values
    y2 = df_agg['woe'].values
    bar_width = 0.5
    axe1.bar(x, y1, color='g', label='COUNT')
    axe1.set_xlabel('GROUP INTERVAL')
    axe1.set_ylabel('COUNT', color='g')
    axe2.plot(x, y2, color='r', label='WOE')
    axe2.set_ylabel('WOE', color='r')

    x_name = (str('(' + str(i.left) + ',' + str(i.right) + ']') for i in list(df_agg[var + '_lab']))
    plt.xticks(x + bar_width / 2, x_name)
    plt.title(var + '_lab')
    plt.show()
    return True


def var_iv_rank(df, var, label, n_group, cnt):
    bins = var_class(df, 'AMT_INCOME_TOTAL', 6, 'TARGET', cnt)

    bin_1 = var_merge(df, var, bins[6][0], 'TARGET')

    bins1 = bin_1[var][0]
    plot_woe(Cal_woe_iv(df, var, bins1, label), var)
    return {'bins': bins1, 'iv': bin_1[var][1]}


if __name__ == '__main__':
    ## 数据加载
    path = "datas/demage.csv"
    df = pd.read_csv(path)

    woe_transfer(df, 'checking')
