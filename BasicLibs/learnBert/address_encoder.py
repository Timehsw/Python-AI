# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/19
    Desc : 
    Note : bert-serving-start -model_dir /tmp/chinese_L-12_H-768_A-12 -num_worker=4
'''
from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
path = '/mnt/d/地址/NLp/deal/poi_area_flat.txt'
df = pd.read_csv(path, sep='\t')

print(df.head())

selected = ['province', 'city', 'district']
df1 = df[selected]
data = df1['province'] + df1['city'] + df1['district']
print(data.head())

# data.to_csv("./add.csv")

print('start encoding ...')
