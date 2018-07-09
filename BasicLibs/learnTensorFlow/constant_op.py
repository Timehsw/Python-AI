# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/8
    Desc : constant操作
'''

import tensorflow as tf

# 1. 定义常量矩阵a和矩阵b
a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2])

print(type(a))
print(type(b))

# 2. 以a和b作为输入,进行矩阵乘法(matmul)操作
c = tf.matmul(a, b)

print(type(c))

print('变量a是否在默认图中:{}'.format(a.graph is tf.get_default_graph()))

# graph = tf.Graph()
# with graph.as_default():
#     # 此时在这个代码块中,使用的就是新定义的图graph
#     # 相当于把默认图换成了graph
#     # 但是只在这个代码块中有效
#     d = tf.constant(5.0)
#     print('变量d是否在新图graph中:{}'.format(d.graph is graph))
#
#     print(d.graph is tf.get_default_graph())
#
#     pass
#
# # 但是只在这个代码块中有效,此时输出为false
#
# print(d.graph is tf.get_default_graph())
#
# with tf.Graph().as_default() as g2:
#     e = tf.constant(5.0)
#     print('变量e是否在新图g2中:{}'.format(e.graph is g2))



# 3. 以a和c作为输入,进行矩阵的加法操作
g = tf.add(a, c, name='add')
print(type(g))
print(g)

# 4.添加减法
h = tf.subtract(b, a, name='b-a')
l = tf.matmul(h, c)
r = tf.add(g, l)


# 会话构建启动图
# 默认情况下创建的session属于默认图
session = tf.Session(graph=tf.get_default_graph())

print(session)

# 调用session的run方法来执行矩阵的乘法,得到c的结果值(所以将c作为参数传递进去)
# 不需要考虑图中间的运行,在运算的时候只需要关注最终结果对应的对象以及所需要的输入数据值
# 只需要传递进去所需要得到的结果对象,会自动的根据图中的依赖关系触发所有相关的OP操作的执行
# 如果OP之间没有依赖关系,tensorflow底层会并行的执行op(有资源的情况下,自动进行)

result = session.run(c)
result1 = session.run(r)
print('type:{},\nvalue:{}'.format(type(result), result))
print('type:{},\nvalue:{}'.format(type(result1), result1))

# 也可以一起传递进行,并且一起计算的时候,不会重复计算重复的地方,可以提高性能
result2 = session.run(fetches=[c, r])
print('type:{},\nvalue:{}'.format(type(result2), result2))

# 会话关闭后,就不可以再使用了
session.close()

# 使用with语句块,会在with语句块执行完成后,自动的关闭Session
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session2:
    print(session2)
    # 获取张量c的结果:通过session的run方法获取
    print("session2 run:{}".format(session2.run(c)))
    # 获取张量c的结果
    print("c eval:{}".format(c.eval()))
