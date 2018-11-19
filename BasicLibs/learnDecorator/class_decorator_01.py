# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/19
    Desc : 类装饰器
    Note : 
'''

'''
没错，装饰器不仅可以是函数，还可以是类，相比函数装饰器，类装饰器具有灵活度大、高内聚、封装性等优点。
使用类装饰器主要依靠类的__call__方法，当使用 @ 形式将装饰器附加到函数上时，就会调用此方法。

使用装饰器极大地复用了代码，但是他有一个缺点就是原函数的元信息不见了，比如函数的docstring、__name__、参数列表
'''

class Foo(object):
    def __init__(self,func):
        self._func=func

    def __call__(self, *args, **kwargs):
        print('class decorator running')
        self._func()
        print('class decorator ending')


@Foo
def bar():
    print('bar')

if __name__ == '__main__':
    bar()