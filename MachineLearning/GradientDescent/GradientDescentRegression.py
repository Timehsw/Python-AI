# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/2
    Desc : 运用梯度下降法求解线性回归问题
    重要,值得好好理解.
'''
import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def validate(X, Y):
    '''
    校验X和Y格式是够正确
    :param X:
    :param Y:
    :return:
    '''

    if len(X) != len(Y):
        raise Exception("样本数与目标值数不一致!")
    else:
        n = len(X[0])
        for l in X:
            if len(l) != n:
                raise Exception("样本数的特征个数不对")

        if len(Y[0]) != 1:
            raise Exception("目标值的属性超过了1个")


def predict(x, theta, intercept=0.0):
    '''
    计算预测值,线性回归
    :param x: 代表其中一条样本
    :param theta:
    :param intercept:
    :return:
    '''

    result = 0.0

    # 1. x和theta相乘
    n = len(x)
    for i in range(n):
        result += x[i] * theta[i]

    # 2. 加入截距项
    result += intercept

    # 3. 返回结果
    return result


def fit(X, Y, alpha=0.01, max_iter=100, fit_intercept=True, tol=1e-4):
    '''
    进行模型训练,返回模型训练得到的theta值
    :param X: 输入的特征矩阵X,要求是一个二维的数组形式;m*n : m表示样本数目,n表示维度数目
    :param Y: 输入的目标矩阵Y,要求是一个二维数组形式;m*k : m表示样本数目,k表示y的值,一般情况为1,这里现阶段也只考虑一个的情况
    :param alpha: 梯度下降中的学习率(步长),默认0.01
    :param max_iter: 梯度下降求解过程中的最大迭代次数,默认100
    :param fit_intercept: 在模型训练过程中是否训练截距项,默认训练
    :param tol: 当损失函数的误差值小于给定值的时候结束循环,默认结束训练,默认1e-4
    :return:
    '''

    # 1.数据校验,校验数据的格式是否正确
    validate(X, Y)

    # 2.开始模型参数计算
    # 获取行和列,分别记做为样本数m和特征属性数目n
    m, n = np.shape(X)

    # 参数定义
    theta = [0 for i in range(n)]
    # 截距项
    intercept = 0
    # 定义临时变量
    diff = [0 for i in range(m)]

    max_iter = 100 if max_iter <= 10 else max_iter

    # 开始进行遍历12733
    for i in range(max_iter):
        # BGD (批量梯度下降法)
        # 在当前theta的取值情况下,预测值和实际值之间的差距
        # 遍历m个样本,将m个样本的真实值与预测值之间的差值存储到diff数组中
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)

            diff[k] = y_true - y_predict

        # 对每一个theta值进行遍历求解,因此遍历每一个特征(维度)
        for j in range(n):
            # 求解出梯度值
            gd = 0
            for k in range(m):
                gd += diff[k] * X[k][j]

            # 进行theta值的更新操作
            theta[j] += alpha * gd

        # 对截距项进行遍历求解(相当于求解theta的时候,对应的维度上的x的取值全部为1)
        if fit_intercept:
            gd = np.sum(diff)
            # 更新截距项
            intercept += alpha * gd

        # 需要判断损失函数现在是否收敛了(损失函数的值小于给定值)
        # 1. 计算损失函数的值
        sum_j = 0.0
        for k in range(m):
            y_true = Y[k][0]
            y_predict = predict(X[k], theta, intercept)
            j = y_true - y_predict
            sum_j += math.pow(j, 2)
        sum_j /= m

        # 2. 当损失函数的值小于给定值的时候,直接结束循环
        if sum_j < tol:
            break

    # 3. 参数返回
    return (theta, intercept)


def predict_X(X, theta, intercept=0.0):
    '''
    对X矩阵进行预测,最终结果返回一个向量
    :param X:
    :param theta:
    :param intercept:
    :return:
    '''
    Y = []
    for x in X:
        Y.append(predict(x, theta, intercept))
    return Y


def score(Y, Y_predict):
    # 1.计算RSS和TSS
    average_y = np.average(Y)
    m = len(Y)
    rss = 0.0
    tss = 0.0

    for k in range(m):
        rss += math.pow(Y[k] - Y_predict[k], 2)
        tss += math.pow(Y[k] - average_y, 2)

    # 2. 计算R^2
    r_2 = 1.0 - 1.0 * rss / tss

    # 3. 返回最终结果
    return r_2


def score_X_Y(X, Y, theta, intercept=0.0):
    Y_predict = predict_X(X, theta, intercept)
    return score(Y, Y_predict)


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(linewidth=1000, suppress=True)
    N = 10
    x = np.linspace(0, 6, N) + np.random.randn(N)
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1

    print(x)
    print(y)

    print("开始模型预测")
    model=LinearRegression(fit_intercept=True)
    model.fit(x,y)
    score(y,model.predict(x))

    theta, intercept = fit(x, y, alpha=0.01, max_iter=100, fit_intercept=True)
    print("参数列表:", theta)
    print("截距项:", intercept)
