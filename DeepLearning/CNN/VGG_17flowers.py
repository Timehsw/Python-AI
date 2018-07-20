# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/20
    Desc : 用VGG网络做17种花数据分类
    Note : 这一份数据集目前在tflearn这个框架中默认自带(pip install tflearn)
'''

from tflearn.datasets import oxflower17
import tensorflow as tf

X, Y = oxflower17.load_data(dirname="./data/17flowers", one_hot=True)
print(X.shape)  # (1360, 224, 224, 3)
print(Y.shape)  # (1360, 17)

# 相关的参数,超参数的设置
# 学习率,一般学习率设置的比较小
learn_rate = 0.1
# 每次迭代的训练样本数量
batch_size = 64
# 训练迭代次数(每个迭代次数中必须训练完一次所有的数据集)
train_epoch = 1000
# 展示信息的间隔大小
display_step = 1

# 开始模型构建
# 1. 设置数据输入的占位符
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, 17], name='y')


def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    '''
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initializer:
    :return:
    '''
    return tf.get_variable(name, shape, dtype, initializer)


# 网络构建
def vgg_network(x, y):
    # cov3-64
    with tf.variable_scope('net1'):
        net1_kernel_size = 64
        net = tf.nn.conv2d(x, filter=get_variable('w', [3, 3, 3, net1_kernel_size]), strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernel_size]))
        net = tf.nn.relu(net)
        # lrn(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)
        # 做一个局部响应归一化，是对卷积核的输出值做归一化
        # depth_radius ==> 对应ppt公式上的n，bias => 对应ppt公式上的k, alpha => 对应ppt公式上的α, beta=>对应ppt公式上的β
        net = tf.nn.lrn(net)
        pass
    pass
