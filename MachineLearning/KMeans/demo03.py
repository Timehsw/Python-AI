# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/18
    Desc : kmean 概率密度图
    Note : 
'''

import pandas as pd
from sklearn.cluster import KMeans #导入K均值聚类算法
from sklearn.cluster import AgglomerativeClustering #层次聚类
from sklearn.cluster import DBSCAN #具有噪声的基于密度聚类方法
import matplotlib.pyplot as plt  #matplotlib画图
from sklearn.decomposition import PCA  #主成分分析
from mpl_toolkits.mplot3d import Axes3D #画三维图
from sklearn.datasets import make_blobs # 导入产生模拟数据的方法


pltcolor=['b<', 'g>', 'r1', 'c2', 'm3', 'y4', 'ks', 'wp']   #颜色
inputfile = 'E:/Work/python/data.xlsx'  #待聚类的数据文件，需要进行标准化的数据文件；
zscoredfile = 'E:/Work/python/zdata.xlsx'  #标准差计算后的数据存储路径文件；

# 1. 产生模拟数据
N = 1000
centers = 4
X, Y = make_blobs(n_samples=N, n_features=4, centers=centers, random_state=28)
data=pd.DataFrame(X,columns=list('abcd'))

iteration = 5000  # 聚类最大循环数
d = []
mink = 4  #聚类的类别范围K值下界
maxk = mink + 1 #聚类的列别范围上界


# svd_solver : string {'auto', 'full', 'arpack', 'randomized'}  这个参数可以在PCA算法里看到
pca = PCA(n_components=2, svd_solver='full')  # 输出两维 PCA主成分分析抽象出维度



newData = pca.fit_transform(data) # 载入N维

if __name__ == '__main__' :
    print("Kmeans")
    for k in range(mink, maxk):  # k取值在mink-maxk值之间，做kmeans聚类，看不同k值对应的簇内误差平方和

        # 读取数据并进行聚类分析
        # 调用k-means算法，进行聚类分析

        kmodel = KMeans(n_clusters=k, init='k-means++', n_jobs=5, max_iter=iteration)  # n_jobs是并行数，一般等于CPU数较好，max_iter是最大迭代次数,init='K-means++'参数的设置可以让初始化均值向量的时候让这几个簇中心尽量分开
        kmodel.fit(data)  # 训练模型
        r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
        #r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心

        print(r1)
        kmlabels = kmodel.labels_  # 得到类别，label_ 是内部变量

        #print(pd.Series(kmlabels, index=data.index)) #输出为序号和对应的类别，以序号为拼接列，把类别加到最后面一列
        r = pd.concat([data, pd.Series(kmlabels, index=data.index)], axis=1)  # 详细输出每个样本对应的类别，横向连接（0是纵向），得到聚类中心对应的类别下的数目
        r.columns = list(data.columns) + [u'聚类类别']  # 重命名表头  加一列的表头
        # r.to_excel(outputfile)  # 保存分类结果

        d.append(kmodel.inertia_)  # inertia簇内误差平方和

        if mink >= maxk - 1:
            x = []
            y = []
            for i in range(0, len(r1)):
                x.append([])
                y.append([])

            for i in range(0, len(kmlabels)):
                labelnum = kmlabels[i] #输出每条数据的类别标签
                if labelnum >= 0:
                    x[labelnum].append(newData[i][0])
                    y[labelnum].append(newData[i][1])

            # blue,green,red,cyan,magenta,yellow,black,white
            for i in range(0, len(r1)):
                plt.plot(x[i], y[i], pltcolor[i])
            plt.show()

    if mink < maxk - 1 :
        plt.plot(range(mink, maxk), d, marker='o')
        plt.xlabel('number of clusters')
        plt.ylabel('distortions')
        plt.show()


    def density_plot(data): #自定义作图函数
        p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
        [p[i].set_ylabel('density') for i in range(k)]
        plt.legend()
        return plt

    pic_output = './datas/' #概率密度图文件名前缀
    for i in range(k):
        print(u'%s%s.png' %(pic_output, i))
        if i > 1:
            density_plot(data[r[u'聚类类别']==i]).savefig(u'%s%s.png' %(pic_output, i))
