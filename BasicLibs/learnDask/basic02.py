# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/12
    Desc : 
    Note : 
'''

from dask.distributed import Client, progress
from dask_ml.wrappers import ParallelPostFit
# Scale up: connect to your own cluster with bmore resources
# see http://dask.pydata.org/en/latest/setup.html
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')
print(client)

import numpy as np
import dask.array as da
from sklearn.datasets import make_classification

X_train, y_train = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1, n_samples=1000)
print(X_train[:5])

# Scale up: increase N, the number of times we replicate the data.
N = 100
X_large = da.concatenate([da.from_array(X_train, chunks=X_train.shape)
                          for _ in range(N)])
y_large = da.concatenate([da.from_array(y_train, chunks=y_train.shape)
                          for _ in range(N)])
X_large
