# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/23
    Desc : 
    Note : 
'''

import pandas as pd
import numpy as np

df=pd.DataFrame({
        'gid':[2,3,1,4,5],
        'data':range(1,10,2)})
    
df=df.set_index('gid')



print(df)
print('-'*40)
df=df.reindex([1,2,3,4,5])
print(df)
#df1=pd.DataFrame({
#        'gid':[1,2,3,4,5],
#        'data':range(10,20,2)})
#df1=df1.set_index('gid')
#print(df1)
#
#df['data2']=df1
#print(df)