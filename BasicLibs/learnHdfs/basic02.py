# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/4
    Desc : 
    Note : 
'''


import os
# Identify the operating system
import platform
import sys
import time
import uuid


import numpy as np
import pandas as pd

#from multiprocessing import Process



from pywebhdfs.webhdfs import PyWebHdfsClient


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

hdfs = PyWebHdfsClient(host='10.111.32.12',port='50070',user_name='dp')

testFile = hdfs.read_file('/user/dp/file/data/hushiwei/datasource/1880/yiumv7d2hbshvyd2ykv1qpaex8ftjdjv.csv')
part_df = pd.read_csv(testFile, index_col=0)
print(part_df.head())
print(mem_usage(part_df))

