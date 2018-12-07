# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/4
    Desc :
    Note :
hdfs dfs -getmerge hdfs://10.111.32.12:8020/user/dp/file/data/yangyang/82/1722/3ecb719e-2ca8-49a2-be24-8a21833ba40c/missing_value/output yangyang_test.csv
'''

import pandas as pd
from hdfs import InsecureClient
import os


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


client_hdfs = InsecureClient('http://' + '10.111.32.12' + ':50070', user='dp')

path = "/user/dp/file/data/yangyang/82/1722/3ecb719e-2ca8-49a2-be24-8a21833ba40c/missing_value/output/"


result_df = []


def read_csv(path):
    parts = client_hdfs.parts(path)
    paths = [path + part for part in parts]
    tmp = []
    for i, part_path in enumerate(paths):
        # ====== Reading files ======
        with client_hdfs.read(part_path, encoding='utf-8') as reader:
            part_df = pd.read_csv(reader, index_col=0)
            tmp.append(part_df)
            print("part %s mem usage %s" % (i, mem_usage(part_df)))

    df = pd.concat(tmp)
    return df


df = read_csv(path)
print("finally mem usage ", mem_usage(df))
