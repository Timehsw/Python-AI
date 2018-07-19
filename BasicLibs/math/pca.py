# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/7/19
    Desc : pca降维,代码实现
    作用 : PCA（principle component analysis）,主成分分析，主要是用来降低数据集的维度，然后挑选出主要的特征
'''

'''
pca思想:

主要思想：移动坐标轴，将n维特征映射到k维上（k<n），这k维是全新的正交特征。这k维特征称为主元，是重新构造出来的k维特征，而不是简单地从n维特征中去除其余n-k维特征。

PCA选择样本点投影具有最大方差的方向


基本步骤：

1. 对数据进行归一化处理（代码中并非这么做的，而是直接减去均值）
2. 计算归一化后的数据集的协方差矩阵
3. 计算协方差矩阵的特征值和特征向量
4. 保留最重要的k个特征（通常k要小于n），也可以自己制定，也可以选择一个阈值，然后通过前k个特征值之和减去后面n-k个特征值之和大于这个阈值，则选择这个k
5. 找出k个特征值对应的特征向量
6. 将m * n的数据集乘以k个n维的特征向量的特征向量（n * k）,得到最后降维的数据。

其实PCA的本质就是对角化协方差矩阵。有必要解释下为什么将特征值按从大到小排序后再选。首先，要明白特征值表示的是什么？在线性代数里面我们求过无数次了，那么它具体有什么意义呢？对一个n*n的对称矩阵进行分解，我们可以求出它的特征值和特征向量，就会产生n个n维的正交基，每个正交基会对应一个特征值。然后把矩阵投影到这N个基上，此时特征值的模就表示矩阵在该基的投影长度。特征值越大，说明矩阵在对应的特征向量上的方差越大，样本点越离散，越容易区分，信息量也就越多。因此，特征值最大的对应的特征向量方向上所包含的信息量就越多，如果某几个特征值很小，那么就说明在该方向的信息量非常少，我们就可以删除小特征值对应方向的数据，只保留大特征值方向对应的数据，这样做以后数据量减小，但有用的信息量都保留下来了。PCA就是这个原理。
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 定义一个均值函数
def meanX(dataX):
    # axis=0表示按照列来求均值
    return np.mean(dataX, axis=0)


def pca(XMat, k):
    '''
    pca方法
    :param XMat: 传入的是一个numpy的矩阵格式,行表示样本数,列表示特征
    :param k: 表示取前k个特征值对应的特征向量
    :return:
    :finalData: 参数一指的是返回的低维矩阵，对应于输入参数二
    :reconData: 参数二对应的是移动坐标轴后的矩阵
    '''
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    # 计算协方差矩阵
    covX = np.cov(data_adjust.T)
    # 求解协方差矩阵的特征值和特征向量
    featValue, featVec = np.linalg.eig(covX)
    # 按照featValue进行从大到小排序
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print('k must lower than feature number')
        return
    else:
        # 注意特征向量的列向量,而numpy的二维矩阵a[m][n]中,a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里需要转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


# 输入文件的每行数据都以\t隔开
def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=-1)).astype(np.float)


# 可视化结果

def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.savefig("outfile.png")
    plt.show()


# 根据数据集data.txt
def main():
    datafile = "./datas/sample.txt"
    XMat = loaddata(datafile)
    k = 2
    return pca(XMat, k)


if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)
