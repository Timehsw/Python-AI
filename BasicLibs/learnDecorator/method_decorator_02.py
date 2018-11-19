# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/19
    Desc : python装饰器---函数带参数
    Note : 
'''
import logging


def use_logging(func):
    def wrapper(*args, **kwargs):
        # args是一个数组，kwargs一个字典
        logging.warn("%s is running" % func.__name__)
        return func(*args, **kwargs)

    return wrapper


@use_logging
def func(name, age=None):
    print("I am %s , age is %s" % (name, age))


if __name__ == '__main__':
    func("tom", 25)
