# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/4
    Desc : 
'''

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from Projects.music_category import feature


def load_model(model_path=None):
    '''
    加载模型
    :param model_path:
    :return:
    '''
    clf = joblib.load(model_save_path)
    return clf


def get_song_feature(path):
    '''
    获取歌曲的分类标签
    :param path:
    :return:
    '''
    data = pd.read_csv(path, header=None, encoding='utf-8')
    name_label_list = np.array(data).tolist()
    index_label_dict = dict(map(lambda t: (t[1], t[0]), name_label_list))
    return index_label_dict


def predict(path, clf, X):
    label_index = clf.predict([X])
    index_label_dict = get_song_feature(path)
    label = index_label_dict[label_index[0]]
    return label


if __name__ == '__main__':
    music_label_path = './datas/music_index_label.csv'
    model_save_path = './datas/music_model.pkl'

    clf = load_model(model_save_path)

    single_song_path = './datas/test/Beyond - 大地.mp3'  # 怀旧
    song_feature = feature.get_single_song_feature(single_song_path)
    label = predict(music_label_path, clf, song_feature)
    print('预测标签为: %s' % label)
