# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/3
    Desc : 
'''


def get(res, tes):
    # 精确度
    n = len(res)
    truth = (res == tes)
    pre = 0
    for flag in truth:
        if flag:
            pre += 1
    return (pre * 100) / n
