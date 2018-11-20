# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/11/20
    Desc : super()
    Note : 当父类和子类均有__int__()方法的时候,子类的__int__方法会覆盖掉父类的,所以如果你在子类中调用了父类的构造方法中的变量,但是
    子类中并没有这个变量的时候,会报错.除非你显示的用super().__int__去初始化父类的方法才行.
'''


class FooParent(object):
    def __init__(self,name):
        self.name=name
        self.parent = 'I\'m the parent.'
        print('Parent')

    def bar(self, message):
        print("%s from Parent" % message)

    def fun(self):
        print('funcs in parents %s' % self.name)


class FooChild(FooParent):
    def __init__(self,name,age):
        self.name=name
        self.age=age
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类B的对象 FooChild 转换为类 FooParent 的对象
        super(FooChild, self).__init__(name)
        self.fun()
        print('Child')

    def bar(self, message):
        super(FooChild, self).bar(message)
        print('Child bar fuction')
        print(self.parent)


if __name__ == '__main__':
    fooChild = FooChild('Tom',23)
    fooChild.bar('HelloWorld')