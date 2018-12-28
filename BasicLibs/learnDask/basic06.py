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
from dask_yarn import YarnCluster
from dask.distributed import Client

from dask_xgboost import XGBClassifier
import dask.dataframe as dd

if __name__ == '__main__':
    # Create a cluster where each worker has two cores and eight GiB of memory
    cluster = YarnCluster(environment='/Users/hushiwei/Downloads/my-yarn-dask.tar.gz',
                          worker_vcores=2,
                          worker_memory="2GiB")

    # Connect to the cluster
    client = Client(cluster)
    print(client.get_versions(check=True))

    # from dask_xgboost import XGBClassifier
    # import dask.dataframe as dd
    #
    # df = dd.read_csv("./datas/Iris.csv")
    #
    # train = df
    # model = XGBClassifier()
    # model.fit(df.iloc[:, :-1].values, df.iloc[:, -1].values)
    # x = model.predict_proba(df.iloc[:, :-1].values)
    # print(x)



