# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/7
    Desc : dump和load
    Note : 这个是持久化到文件中
'''

import pickle

dic = {
    'spark': 10,
    'hadoop': 9,
    'flume': 2
}

# 将 obj 持久化保存到文件 tmp.txt 中
# pickle.dump(dic, open("./tmp/dic.txt", "wb+"))

# do something else ...

# 从 tmp.txt 中读取并恢复 obj 对象
obj2 = pickle.load(open("./tmp/dic.txt", "rb"))
print(obj2)
print(type(obj2))
