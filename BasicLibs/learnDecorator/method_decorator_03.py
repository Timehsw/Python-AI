# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/19
    Desc : python装饰器---装饰器和函数都带参数
    Note : 
'''
import logging


def use_logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == "warn":
                logging.warn("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


@use_logging(level="warn")
def foo(name='foo'):
    print("i am %s" % name)


if __name__ == '__main__':
    foo()
