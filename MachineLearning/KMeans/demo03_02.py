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


pltcolor=['b<', 'g>', 'r1', 'c2', 'm3', 'y4', 'ks', 'wp']   #颜色
inputfile = 'E:/Work/python/data.xlsx'  #待聚类的数据文件，需要进行标准化的数据文件；
zscoredfile = 'E:/Work/python/zdata.xlsx'  #标准差计算后的数据存储路径文件；

data = pd.read_excel(inputfile)  #读取数据，因为我的文件时CSV格式的，直接read_excel就可以，不是的话自己写一个文件读取的方法读进来。
iteration = 5000  # 聚类最大循环数
d = []
mink = 4  #聚类的类别范围K值下界
maxk = mink + 1 #聚类的列别范围上界

#data = (data - data.mean(axis = 0))/(data.std(axis = 0)) #简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。也可以直接在原始数据里面就先处理好，这句话就可以不用了。
#data.columns=['Z'+i for i in data.columns] #表头重命名。
#data.to_excel(zscoredfile, index = False) #数据写入，标准化之后数据重新写入，直接覆盖掉之前的文件里的内容


# svd_solver : string {'auto', 'full', 'arpack', 'randomized'}  这个参数可以在PCA算法里看到
pca = PCA(n_components=2, svd_solver='full')  # 输出两维 PCA主成分分析抽象出维度

#下面注释的几行是其它几种PCA主成分分析方法，可以尝试使用。
# from sklearn.decomposition import KernelPCA
# kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
# pca = KernelPCA(n_components = 2, kernel = "poly")
# method : {'lars', 'cd'}

# from sklearn.decomposition import SparsePCA
# pca = SparsePCA(n_components = 2, method = 'cd')

# from sklearn.decomposition import TruncatedSVD
# algorithm : string, default = "randomized" "arpack"
# pca = TruncatedSVD(algorithm = "arpack")

newData = pca.fit_transform(data) # 载入N维

if __name__ == "__main__3" :
    print("DBSCAN")

    outputfile = 'E:/Work/python/fenlei_DBSCAN.xlsx'
    # 读取数据并进行聚类分析
    # 调用DBSCAN算法，进行聚类分析
    linkages = ['ward', 'average', 'complete']

    #From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. These metrics support sparse matrix inputs.
    #From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    kmodel = DBSCAN(eps= 0.768, min_samples=160, n_jobs=7, metric='euclidean')
    # kmodel = DBSCAN(eps=0.64, min_samples=100, n_jobs=7, metric='canberra')

    kmodel.fit(data)  # 训练模型
    r1 = pd.Series(kmodel.labels_).value_counts()  # 统计各个类别的数目
    print(r1)
    # r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心

    kmlabels = kmodel.labels_
    r = pd.concat([data, pd.Series(kmlabels, index=data.index)], axis=1)  # 详细输出每个样本对应的类别， 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
    print(outputfile)
    r.to_excel(outputfile)  # 保存分类结果

    rlen = len(r1)
    if (rlen <= 6):
        x = []
        y = []
        for i in range(0, len(r1) + 1):
            x.append([])
            y.append([])
        for i in range(0, len(kmlabels)):
            labelnum = kmlabels[i] + 1
            if labelnum >= 0:
                x[labelnum].append(newData[i][0])
                y[labelnum].append(newData[i][1])
        # blue,green,red,cyan,magenta,yellow,black,white
        for i in range(0, len(r1) + 1):
            plt.plot(x[i], y[i], pltcolor[i])
        plt.show()

