# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/1
    Desc : 用RNN进行文本分类
    Note : 
'''

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from Projects.text_classification_cnn_rnn.rnn_model import TRNNConfig, TextRNN
from Projects.text_classification_cnn_rnn.cnn_model import TCNNConfig, TextCNN

from Projects.text_classification_cnn_rnn.helper.cnews_loader import build_vocab, read_category, read_vocab, \
    process_file

base_dir = 'data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'base_validation')  # 最佳验证结果保存路径


def get_time_diff(start_time):
    '''
    统计用时时长
    :param start_time:
    :return:
    '''
    end_time = time.time()
    time_diff = end_time - start_time
    use_time = timedelta(seconds=int(round(time_diff)))

    print('Time usage: {}'.format(use_time))


def train():
    print('Configuring TensorBoard and Saver ...')
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Loading training and validation data...')
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, words_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, words_to_id, cat_to_id, config.seq_length)
    get_time_diff(start_time)

    # 创建Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升,提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)


def test():
    pass


if __name__ == '__main__':
    print("Configuring Model ... ")
    # config = TRNNConfig()
    config = TCNNConfig()

    # 如何没有词汇表,则从训练集中构建词汇表
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)

    categories, cat_to_id = read_category()
    words, words_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)

    model = TextCNN(config)
    option = 'train'
    if option == 'train':
        train()
    else:
        test()
