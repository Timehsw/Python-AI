# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-02-19
    Desc : 
    Note : 
'''

import pandas as pd

path = "/Users/hushiwei/Downloads/model2/data/test.csv"
df = pd.read_csv(path)
df = df.fillna(-999)
print(df.head())
outpath = "/Users/hushiwei/Downloads/model2/data/test_no_missing.csv"

df.to_csv(outpath, index=False)
