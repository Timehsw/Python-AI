# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/13
    Desc : 
    Note : 先执行 dask-scheduler ,再运行代码
'''

from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
from dask_xgboost import XGBClassifier
import dask.dataframe as dd

if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    import dask

    client = Client(address='10.111.26.209:8786')
    from dask_xgboost import XGBClassifier
    import dask.dataframe as dd

    # from dask_ml.cluster import KMeans
    # print(df)
    # from dask_ml.linear_model import LogisticRegression
    df = dd.read_csv("./datas/Iris.csv")

    train = df
    model = XGBClassifier()
    model.fit(df.iloc[:, :-1].values, df.iloc[:, -1].values)
    x = model.predict_proba(df.iloc[:, :-1].values)
    print(x)



