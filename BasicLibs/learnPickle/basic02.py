# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/7
    Desc : dumps和loads
    Note : 这个不用持久化到文件中,返回为字符串,可以从字符串中恢复
'''

import pickle

dic = {
    'spark': 10,
    'hadoop': 9,
    'flume': 2
}

model = pickle.dumps(dic)
print(model)
res = pickle.loads(model)
print(res)
