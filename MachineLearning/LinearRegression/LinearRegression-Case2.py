# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/5/28
    Desc : 波士顿房屋租赁价格预测
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import sklearn

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
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


def notEmpty(s):
    return s != ""


# 加载数据
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

path='./datas/boston_housing.data'

df=pd.read_csv(path,header=None)
print(df.head())

# 分隔符不对,重新整理数据
# 由于数据文件格式不统一,所以读取的时候,先按照一行一个字段属性读取数据,然后再按照每行数据进行处理
# print(df.shape)
data = np.empty((len(df), 14))
for i,d in enumerate(df.values):
    # i是每行index,d是每行数据
    # 过滤空值,并转换成float类型
    d=map(float,filter(notEmpty,d[0].split(' ')))
    data[i]=list(d)

print(data.shape)

# 分割数据
x,y=np.split(data,(13,),axis=1)
print(x[0:5])

# 转换格式,拉直操作
y=y.ravel()
print(y[0:5])
ly=len(y)
print('样本数据量: %d,特征个数: %d' % x.shape)
print('target样本数据量: %d' % y.shape[0])

# pipeline常用于并行调参
models=[
    Pipeline([
        ('ss',StandardScaler()),
        ('poly',PolynomialFeatures()),
        ('linear',RidgeCV(alphas=np.logspace(-3,1,20)))
    ]),
    Pipeline([
        ('ss',StandardScaler()),
        ('poly',PolynomialFeatures()),
        ('linear',LassoCV(alphas=np.logspace(-3,1,20)))
    ])
]

# 参数字典,字典中的Key是属性的名称,value是可选的参数列表
parameters={
    "poly__degree":[3,2,1],
    "poly__interaction_only":[True,False], #不产生交互项,如X1*X1
    "poly__include_bias":[True,False], # 多项式幂为零的特征作为线性模型中的截距;true表示包含
    "linear__fit_intercept":[True,False]
}

# 训练集测试集划分
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Lasso 和 Ridge 模型比较运行图表展示
titles=['Ridge','Lasso']
colors=['g-','b-']
plt.figure(figsize=(16,8),facecolor='w')
ln_x_test=range(len(x_test))

plt.plot(ln_x_test,y_test,'r-',lw=2,label='真实值')
for t in range(2):
    # 获取模型并设置参数
    # GridSearchCV:进行交叉验证,选择出最优的参数值出来
    # 第一个输入参数: 进行参数选择的模型,
    # param_grid: 用于进行模型选择的参数字段,要求是字典类型
    # cv: 进行几折交叉验证
    model=GridSearchCV(models[t],param_grid=parameters,cv=5,n_jobs=1)
    # 模型训练-网格搜索
    model.fit(x_train,y_train)
    # 模型效果值获取(最优参数)
    print("%s算法:最优参数:%s" % (titles[t],model.best_params_))
    print("%s算法:R值=%.3f" % (titles[t],model.best_score_))

    # 模型预测
    y_predict = model.predict(x_test)

    # 画图
    plt.plot(ln_x_test,y_predict,colors[t],lw=t+3,label='%s算法估计值,R^2=%.3f' %(titles[t],model.best_score_))

# 图形显示
plt.legend(loc='upper left')
plt.grid(True)
plt.title('波士顿房屋价格预测')
plt.show()

# 模型训练---> 单个Lasso模型(一阶特征选择)

model=Pipeline([
    ('ss',StandardScaler()),
    ('poly',PolynomialFeatures(degree=1,include_bias=False,interaction_only=True)),
    ('linear',LassoCV(alphas=np.logspace(-3,1,20),fit_intercept=False))
])

# 模型训练
model.fit(x_train,y_train)

# 模型评估
# 数据输出
print('参数:',list(zip(names,model.get_params('linear')['linear'].coef_)))
print('截距:',model.get_params('linear')['linear'].intercept_)