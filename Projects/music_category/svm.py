# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/3
    Desc : 用网格交叉验证来进行模型选择
'''

import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib
from Projects.music_category import acc


def grid_search(X, Y):
    '''
    进行网格交叉验证,返回最好模型
    :param X:
    :param Y:
    :return:
    '''
    parameters = {
        'kernel': ('linear', 'rbf', 'poly'),
        'C': [0.1, 1],
        'probability': [True, False],
        'decision_function_shape': ['ovo', 'ovr']
    }
    clf = GridSearchCV(svm.SVC(random_state=0), param_grid=parameters, cv=5)
    print("开始交叉验证获取最优参数构建")
    clf.fit(X, Y)
    print('最优参数: ', clf.best_params_)
    print('最优模型准确率:', clf.best_score_)


def grid_search_best_model(music_csv_file_path, data_percentage=0.7):
    '''
    网格交叉验证数据准备
    :param music_csv_file_path:
    :param data_percentage:
    :return:
    '''
    print("开始读取数据: " + music_csv_file_path)
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')

    data = data.sample(frac=data_percentage).T
    X = data[:-1].T
    Y = np.array(data[-1:])[0]
    # print(Y)
    grid_search(X, Y)


def polynomial_model(X, Y):
    '''
    将网格交叉验证得出的最优模型参数传入模型中
    :param X:
    :param Y:
    :return:
    '''
    clf = svm.SVC(kernel='poly', C=0.1, probability=True, decision_function_shape='ovo', random_state=0)
    clf.fit(X, Y)
    # score = clf.score(X, Y)
    score = acc.get(clf.predict(X), Y)
    # 返回模型及预测准确度
    return clf, score


def muti_train_model_save(train_percentage=0.7, fold=1, music_csv_file_path=None, model_out_f=None):
    '''
    多次训练模型,将最优的模型保存下来
    :param train_percentage:
    :param fold:
    :param music_csv_file_path:
    :param model_out_f:
    :return:
    '''
    data = pd.read_csv(music_csv_file_path, sep=',', header=None, encoding='utf-8')

    max_train_score = None
    max_test_score = None
    max_score = None
    best_clf = None
    flag = True
    for index in range(1, int(fold) + 1):
        print(index)
        shuffle_data = shuffle(data)
        X = shuffle_data.T[:-1].T
        Y = np.array(shuffle_data.T[-1:])[0]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_percentage)

        (clf, train_score) = polynomial_model(x_train, y_train)
        # test_score = clf.score(x_test, y_test)
        test_score = acc.get(clf.predict(x_test), y_test)

        # 模型综合准确率
        score = 0.35 * train_score + 0.65 * test_score
        if flag:
            max_score = score
            max_train_score = train_score
            max_test_score = test_score
            best_clf = clf
            flag = False
        else:
            if max_score < score:
                max_score = score
                max_train_score = train_score
                max_test_score = test_score
                best_clf = clf
        print('第%d次训练，训练集上的正确率为：%0.2f, 测试集上正确率为：%0.2f,加权平均正确率为：%0.2f' % (index, train_score, test_score, score))

    print('最优模型效果：训练集上的正确率为：%0.2f,测试集上的正确率为：%0.2f, 加权评卷正确率为：%0.2f' % (max_train_score, max_test_score, max_score))

    print('最优模型是:', best_clf)
    joblib.dump(best_clf, model_out_f)


if __name__ == '__main__':
    music_label_path = './datas/music_index_label.csv'
    music_feature_save_path = './datas/music_feature.csv'

    model_save_path = './datas/music_model.pkl'

    print('-' * 30, "网格训练寻找最合适模型开始...", '-' * 30)
    start = time.time()
    grid_search_best_model(music_feature_save_path)
    end = time.time()
    print('寻找最佳模型共耗时%.2f' % (end - start))

    print('-' * 30, "网格训练寻找最合适模型开始...", '-' * 30)
    start = time.time()
    muti_train_model_save(train_percentage=0.7, fold=1000, music_csv_file_path=music_feature_save_path,
                          model_out_f=model_save_path)
    end = time.time()
    print('训练模型共耗时%.2f' % (end - start))
