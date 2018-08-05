# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/1
    Desc : rnn模型/参数配置/模型层次
    Note : 
'''

import tensorflow as tf


class TRNNConfig(object):
    '''
    RNN模型相关配置参数
    '''

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表大小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10  # 总迭代次数

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard




class TextRNN(object):
    pass
