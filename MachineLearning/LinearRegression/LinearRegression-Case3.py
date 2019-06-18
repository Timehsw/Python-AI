# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/5/28
    Desc : 葡萄酒质量预测
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import sklearn

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

# 读取数据,红葡萄酒type=1和白葡萄酒type=2
path1='./datas/winequality-red.csv'
path2='./datas/winequality-white.csv'
df1=pd.read_csv(path1,sep=";")
df1['type']=1
df2=pd.read_csv(path2,sep=";")
df2['type']=2

# 合并数据
df=pd.concat([df1,df2],axis=0)
print(df['type'].value_counts())
print(df.head())

# 获取因变量与自变量
quality = "quality"

#
names=df.columns.tolist()

xs=list(filter(lambda x:x!=quality,names))


# 异常数据处理

new_df = df.replace('?', np.nan)
datas=new_df.dropna(how='any')

X=datas[xs]
Y=datas[quality]


# 创建模型列表
models=[
    Pipeline([
        ('Poly',PolynomialFeatures()),
        ('Linear',LinearRegression())
    ]),
    Pipeline([
        ('Poly',PolynomialFeatures()),
        ('Linear',RidgeCV(alphas=np.logspace(-4,2,20)))
    ]),
    Pipeline([
        ('Poly',PolynomialFeatures()),
        ('Linear',LassoCV(alphas=np.logspace(-4,2,20)))
    ]),
    Pipeline([
        ('Poly',PolynomialFeatures()),
        ('Linear',ElasticNetCV(alphas=np.logspace(-4,2,20),l1_ratio=np.logspace(0,1,5)))
    ])
]

plt.figure(figsize=(16,8),facecolor='w')
titles=['线性回归预测','Ridge回归预测','Lasso回归预测','ElasticNet预测']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.01,random_state=0)
ln_x_test=range(len(X_test))

# 给定阶数以及颜色
d_pool=np.arange(1,4,1) # 1 2 3 阶
m=len(d_pool)

clrs = [] # 颜色
for c in np.linspace(5570560, 255, m):
    clrs.append('#%06x' % int(c))

for t in range(4):
    plt.subplot(2, 2, t + 1)
    model = models[t]
    plt.plot(ln_x_test, Y_test, c='r', lw=2, alpha=0.75, zorder=10, label=u'真实值')
    for i,d in enumerate(d_pool):
        # 设置参数
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(X_train, Y_train)
        # 模型预测及计算R^2
        Y_pre = model.predict(X_test)
        R = model.score(X_train, Y_train)
        # 输出信息
        lin = model.get_params('Linear')['Linear']
        output = u"%s:%d阶, 截距:%d, 系数:" % (titles[t], d, lin.intercept_)
        print(output, lin.coef_)
        ## 图形展示
        plt.plot(ln_x_test, Y_pre, c=clrs[i], lw=2,alpha=0.75, zorder=i, label=u'%d阶预测值,$R^2$=%.3f' % (d,R))
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.suptitle(u'葡萄酒质量预测', fontsize=22)
plt.show()