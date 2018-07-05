# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/3
    Desc : 
'''

import numpy as np
import pandas as pd
import glob
import time
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile
from python_speech_features import mfcc
import os
import sys


def get_song_list():
    data = pd.read_csv(song_list_path)
    data = data[['name', 'tag']]
    return data


def get_single_song_feature(song_path):
    '''
    抽取出一首歌曲的特征
    :param song_path: 歌曲路径
    :return: 歌曲的特征矩阵
    '''
    song_name_arr = song_path.split('.')
    file_format = song_name_arr[-1].lower()  # 获取歌曲格式
    file_name = song_path[:-(len(file_format) + 1)]  # 获取歌曲名称
    if file_name != 'wav':
        # 把mp3格式转换为wav,保存至原文件夹中
        song = AudioSegment.from_file(song_path, format='mp3')
        file = file_name + ".wav"
        song.export(file, format='wav')
    try:
        rate, data = wavfile.read(file)
        mfcc_feas = mfcc(data, rate, numcep=13, nfft=2048)
        mm = np.transpose(mfcc_feas)
        mf = np.mean(mm, axis=1)
        mc = np.cov(mm)
        result = mf
        for i in range(mm.shape[0]):
            result = np.append(result, np.diag(mc, i))
            # os.remove(file)
        return result
    except Exception as msg:
        print(msg)


def feature_extract():
    df = get_song_list()
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0], t[1]), name_label_list))
    # 抽取歌曲标签,进行去重操作
    labels = set(name_label_dict.values())
    # 对歌曲标签进行数值化处理,返回歌曲标签数值映射
    label_index_dict = dict(zip(labels, np.arange(len(labels))))
    all_music_files = glob.glob(song_path)
    # all_music_files中是所有的mp3文件,然后进行排序
    all_music_files.sort()

    loop_count = 0
    flag = True
    all_mfcc = np.array([])

    for file_name in all_music_files:
        # ./datas/music/方大同 - 三人游.mp3
        print("开始处理: ", file_name)
        music_name = file_name.split('/')[-1].split('.')[-2].split('-')[-1].strip()

        if music_name in name_label_dict:
            # 返回样本标签数值化
            label_index = label_index_dict[name_label_dict[music_name]]
            ff = get_single_song_feature(file_name)
            ff = np.append(ff, label_index)

            if flag:
                all_mfcc = ff
                flag = False
            else:
                all_mfcc = np.vstack([all_mfcc, ff])
        else:
            print('无法处理：' + file_name.replace('\xa0', '') + '; 原因是：找不到对应的label')
        print('looping --- %d' % loop_count)
        print('all_mfcc.shape: ', end='')
        print(all_mfcc.shape)
        loop_count += 1
    # 保存数据
    label_index_list = []
    for k in label_index_dict:
        label_index_list.append([k, label_index_dict[k]])

    pd.DataFrame(label_index_list).to_csv(music_label_path, header=None, index=False, encoding='utf-8')
    pd.DataFrame(all_mfcc).to_csv(music_feature_save_path, header=None, index=False, encoding='utf-8')
    return all_mfcc


if __name__ == '__main__':
    song_list_path = "./datas/music_info.csv"
    song_path = './datas/music/*.mp3'
    music_label_path = './datas/music_index_label.csv'
    music_feature_save_path = './datas/music_feature.csv'

    start = time.time()

    feature_extract()

    end = time.time()

    print('特征处理总耗时:%.2fs' % (end - start))
