# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/19
    Desc : python装饰器
    Note : 
'''
import logging



def use_logging(func):

    def wrapper():
        # args是一个数组，kwargs一个字典
        logging.warn("%s is running" % func.__name__)
        return func()
    return wrapper


@use_logging
def func():
    print("func")



if __name__ == '__main__':
    func()