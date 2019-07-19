# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/18
    Desc : 
    Note : 
'''

import collections
import tensorflow as tf
import random
import numpy as np
import sys

# read data
file_path = './data/belling_the_cat.txt'
content = ""

with open(file_path, 'r') as fp:
    content = fp.read()

words = content.split()


def build_dataset(words):
    count = collections.Counter(words).most_common()
    # 构建正向字典
    dictionary = dict()
    for key, _ in count:
        dictionary[key] = len(dictionary)

    # 构建反向字典
    reserve_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reserve_dictionary


dictionary, reserve_dictionary = build_dataset(words)
vocal_size = len(dictionary)
# 开始构建模型参数
n_input = 3
n_hidden = 512
batch_size = 20
weight = tf.get_variable('weight_out', [2 * n_hidden, vocal_size], initializer=tf.random_normal_initializer)
bias = tf.get_variable('bias_out', [vocal_size], initializer=tf.random_normal_initializer)


# 构建LSTM神经网络
def RNN(x, weight, bias):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)
    # 调用tensorflow接口来定义lstm_cell
    # 构建前向cell
    rnn_cell_format = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1.0)
    # 构建后向cell
    rnn_cell_backmat = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1.0)

    # 构建双向LSTM
    outputs = tf.nn.static_bidirectional_rnn(rnn_cell_format, rnn_cell_backmat,
                                             inputs=x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weight) + bias


# 构造随机词典映射
def build_data(offset):
    while offset + n_input > vocal_size:
        offset = random.randint(0, vocal_size - n_input)
    symbols_in_key = [dictionary[str(words[i])] for i in range(offset, offset + n_input)]
    symbols_out_onehot = np.zeros([vocal_size],dtype=np.float)
    symbols_out_onehot = [dictionary[str(words[offset + n_input])]] = 1.0
    return symbols_in_key, symbols_out_onehot

a,b=build_data(3)
print(a)
print(b)
# 创建softmax交叉熵
x=tf.placeholder(tf.float32,[None,n_input,1])
y=tf.placeholder(tf.float32,[None,vocal_size])
pred=RNN(x,weight,bias)
# 损失函数构造
cost=tf.reduce_mean(tf.nn.softmax(logits=pred,labels=y))
# 优化器构造
optimizer=tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

# 精确率计算

# 创建会话
