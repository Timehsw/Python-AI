# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/20
    Desc : 手写数字识别的CNN网络 LeNet
    Note : 结合SimpleNeuralNetwork代码学习学习
'''

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import sys

tf.set_random_seed(28)

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# 手写数字识别的数据集主要包含三个部分：训练集(5.5w, mnist.train)、测试集(1w, mnist.test)、验证集(0.5w, mnist.validation)
# 手写数字图片大小是28*28*1像素的图片(黑白)，也就是每个图片由784维的特征描述
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
train_sample_number = mnist.train.num_examples  # 55000


def show_image(i):
    '''
    查看第i个样本的值和图片
    :param i:
    :return:
    '''
    text = "true number is: " + str(np.argmax(train_label[i, :]))
    one_pic = train_img[i].reshape((28, 28, 1))
    ax = plt.figure()
    ax.text(0.1, 0.9, text, ha='center', va='center')
    plt.imshow(one_pic[:, :, 0], cmap='Greys_r')
    plt.show()


# show_image(105)

# 定义相关的超参数
learn_rate = 0.01
batch_size = 64
display_step = 1

# 准备x,y
input_dim = train_img.shape[1]
n_classes = train_label.shape[1]
x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')


def get_variable(name='w', shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer)


# 构建LeNet神经网络
def lenet(x, y):
    with tf.variable_scope('input1'):
        net = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.variable_scope('conv2'):
        net = tf.nn.conv2d(net, filter=get_variable('w', shape=[5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', shape=[20]))
        net = tf.nn.relu(net)
    with tf.variable_scope('pool3'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv4'):
        net = tf.nn.conv2d(net, filter=get_variable('w', shape=[5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net,get_variable('b', shape=[50]))
        net = tf.nn.relu(net)
    with tf.variable_scope('pool5'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('fc6'):
        size = 7 * 7 * 50
        net = tf.reshape(net, shape=[-1, size])
        net = tf.add(tf.matmul(net, get_variable('w', shape=[size, 500])), get_variable('b', shape=[500]))
        net = tf.nn.relu(net)
    with tf.variable_scope('fc7'):
        net = tf.add(tf.matmul(net, get_variable('w', shape=[500, n_classes])), get_variable('b', shape=[n_classes]))

    return net


# 网络最后一层的输出
act = lenet(x, y)

# 构建模型的损失函数
# softmax_cross_entropy_with_logits会先计算softmax在计算与真实值的交叉熵值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

# 最小化损失函数
train = tf.train.AdadeltaOptimizer(learning_rate=learn_rate).minimize(cost)

pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()
    epoch = 0
    while True:
        avg_cost = 0
        # 55000/64=859.37=859
        total_batch = int(train_sample_number / batch_size)
        # 迭代更新
        for i in range(total_batch):
            # 获取x和y,填充数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            # 模型训练
            sess.run(train, feed_dict=feeds)
            # 获取损失函数值
            avg_cost += sess.run(cost, feed_dict=feeds)

        avg_cost = avg_cost / total_batch

        # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
        if (epoch + 1) % display_step == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_cost))
            # 这里之所以使用batch_xs和batch_ys，是因为我使用train_img会出现内存不够的情况，直接就会退出
            feeds = {x: train_img[:1000], y: train_label[:1000]}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {x: test_img, y: test_label}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)

            if train_acc >= 0.99 and test_acc >= 0.99:
                saver.save(sess, './mn/model_{}_{}'.format(train_acc, test_acc), global_step=epoch)
                break
        epoch += 1

    # 模型可视化输出
    writer = tf.summary.FileWriter('./mn/graph', tf.get_default_graph())
    writer.close()

print('end....')
