# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/5/28
    Desc : 
'''

def buy(x):
    return x!='2'

arr=['1','2','3','4']

i = filter(buy, arr)
for l in i:
    print(l)
    print(type(l))


re=map(float,filter(buy, arr))

for l in re:
    print(l)
    print(type(l))