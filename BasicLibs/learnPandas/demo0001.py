# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/19
    Desc : 
    Note : 
'''

import pandas as pd

path = '/mnt/d/地址/NLp/deal/poi_area_flat.txt'
df = pd.read_csv(path, sep='\t')

print(df.head())

selected = ['province', 'city', 'district']
df1 = df[selected]
df_right=df1[df1['province']=='江苏省']
# df2.to_csv("./add.csv")


path1 = '/mnt/d/地址/NLp/deal/poi_amap_jiangsu_fil.txt'
df3=pd.read_csv(path1,sep='\t')
df4=df3[['name','address','type']]
print(df3)

df5=df4.merge(df1,left_on='address',right_on='district')
print(df5)
df6=df5[['province','city','address','name','type']]
print(df6)
df6.to_csv('addr.csv',index=False)

