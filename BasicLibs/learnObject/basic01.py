# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/27
    Desc : 
    Note : 
'''


class Father:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print('~' * 10, self.name, '~' * 10)

    def jump(self,name):
        print('I can jump',name)

    def swim(self):
        print('I can swim')


class Son(Father):


    def printSome(self):
        super().show()
        super().jump(self.name)
        super().swim()
        print('---------------========')
        self.show()
        self.jump(self.name)


if __name__ == '__main__':
    son = Son('aaa', 23)
    son.printSome()
    print('---------------')
    son.swim()
