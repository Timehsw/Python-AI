# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/8/3
    Desc : 文本处理类
    Note : 
'''
import sys
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr


def open_file(filename, mode='r'):
    '''
    打开文件,返回文件句柄
    :param filename:
    :param mode:
    :return:
    '''
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    '''
    读取文件数据
    :param filename:
    :return:
    '''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '''
    根据训练集构建词汇表,然后存储到文件中
    :param train_dir:
    :param vocab_dir:
    :param vocab_size:
    :return:
    '''
    data_train, _ = read_file(train_dir)
    # print(len(data_train)) 50000

    # 将50000条数据中的单词进行收集
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    # most_common对单词进行计算,只要最多的指定个数的单词
    count_pairs = counter.most_common(vocab_size - 1)
    # 写法不错.取出单词,不要统计的个数
    words, _ = list(zip(*count_pairs))

    # 在头部拼接上<PAD>
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    print(len(words))
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    '''
    读取词汇表,将词汇数值化
    :param vocab_dir:
    :return:
    '''
    with open_file(vocab_dir) as f:
        words = [_.strip() for _ in f.readlines()]
    words_to_id = dict(zip(words, range(len(words))))
    return words, words_to_id


def read_category():
    '''
    读取分类目录
    :return:
    '''
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''
    将文件转换为id 表示
    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return:
    '''
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append(word_to_id[x] for x in contents[i] if x in word_to_id)
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    '''
    生成批次数据
    :param x:
    :param y:
    :param batch_size:
    :return:
    '''
    pass
