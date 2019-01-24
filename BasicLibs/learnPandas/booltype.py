# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-01-22
    Desc : 
    Note : 
'''

import pandas as pd
import numpy as np

path = './data/demo.csv'
# df = pd.read_csv(path, dtype={'A': np.int64, 'B': np.int64, 'C': np.int64, 'D': np.bool})
df = pd.read_csv(path,na_filter=False)
print(df)
print(df.dtypes)

df2=df.fillna({'C': 0, 'D': False})
print(df2)
print(df2.dtypes)
