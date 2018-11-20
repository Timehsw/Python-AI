# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/20
    Desc : python解包操作
    Note : 
'''


def jieba_func(name='Tom', age=25):
    print(name)
    print(age)


param = {
    'name': 'Jerry',
    'age': 18

}
jieba_func()
jieba_func(**param)
