# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/21
    Desc : tf.app.flags例子
    Note : 
'''

import tensorflow as tf

# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string('str_name', 'def_v_1', "descrip1")
tf.app.flags.DEFINE_string('my_name', 'hushiwei', "name")
tf.app.flags.DEFINE_integer('int_name', 10, "descript2")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")

FLAGS = tf.app.flags.FLAGS


# 必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   main的参数名随意定义，无要求
def main(_):
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)
    print(FLAGS.my_name)


if __name__ == '__main__':
    tf.app.run()  # 执行main函数
