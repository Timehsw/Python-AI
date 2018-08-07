# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/6
    Desc : 卷积神经网络LeNet5
    Note : 
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取mnist数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 注册默认session 后面操作无需指定session 不同sesson之间的数据是独立的
sess = tf.InteractiveSession()


# 构造参数W函数 给一些偏差0.1防止死亡节点
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 构造偏差b函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# x是输入,W为卷积参数 如[5,5,1,30] 前两个表示卷积核的尺寸
# 第三个表示通道channel  第四个表示提取多少类特征
# strides 表示卷积模板移动的步长都是 1代表不遗漏的划过图片每一个点
# padding 表示边界处理方式这里的SAME代表给边界加上padding让输出和输入保持相同尺寸
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# ksize 使用2x2最大池化即将一个2x2像素块变为1x1 最大池化保持像素最高的点
# stride也横竖两个方向为2歩长,如果步长为1 得到尺寸不变的图片
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义张量流输入格式
# reshape变换张量shape 2维张量变4维 [None, 784] to [-1,28,28,1]
# -1表示样本数量不固定 28 28为尺寸 1为通道
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 第一次卷积池化 卷积层用ReLU激活函数
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二次卷积池化 卷积层用ReLU激活函数
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 全连接层使用ReLU激活函数  reshape改变张量结构 变成一维
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 为了减轻过拟合使用一个Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Dropout层 softmax连接输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# loss函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# 优化算法Adam函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# accuracy函数
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
# 训练20000次 每次大小为50的mini-batch 每100次训练查看训练结果 用以实时监测模型性能
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, train_accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
}))
