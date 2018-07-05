# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/4/27.

    desc: 时间与功率之间的关系
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置字符集,防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


# 加载数据
df = pd.read_csv(
    "/Users/hushiwei/PycharmProjects/Python-AI/MachineLearning/LinearRegression/datas/household_power_consumption_1000.txt",
    sep=";")
print(df.head())

# 异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')

# 观察数据

print(datas.describe())


# 处理时间

def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 获取X和Y,并将时间转换为数值型变量
# X:时间
# Y:用电量

X=datas.iloc[:,0:2]
X=X.apply(lambda x:pd.Series(date_format(x)),axis=1)
Y=datas['Global_active_power']


print(X.head())
print(Y.head())


# 划分数据集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# 对数据进行标准化
ss=StandardScaler()

# 训练并转换
X_train=ss.fit_transform(X_train)
# 直接使用在模型构建数据上进行一个数据标准化操作
X_test=ss.transform(X_test)

# 这里,fit_transform和transform的区别?
# fit_tranform是计算并转换,在这个过程中会计算出均值和方差,然后将变量进行标准化去量纲
# transform是转换,它将用上面计算出的均值和方差来进行标准化去量纲.
# 因为训练集的数据相较于测试集更多 ,所以测试集也延用训练集算出来的均值和方差
# 因此fit_transform在transform上面调用

# 模型训练

liner=LinearRegression()
liner.fit(X_train,Y_train)


# 模型校验

y_predict=liner.predict(X_test)
time_name=['年','月','日','时','分','秒']
print("准确率:",liner.score(X_train,Y_train))
mse=np.average((y_predict-Y_test)**2)
rmse=np.sqrt(mse)
print('rmse:',rmse)
print("特征的系数为:",list(zip(time_name,liner.coef_)))

# 模型保存

# from sklearn.externals import joblib
#
# joblib.dump(ss,"data_ss.model") ## 将标准化模型保存
# joblib.dump(liner,"data_lr.model") ## 将模型保存
#
# joblib.load("data_ss.model") ## 加载模型
# joblib.load("data_lr.model") ## 加载模型

# 预测值和实际值可视化画图比较

t=np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t,Y_test,'r-',linewidth=2,label='真实值')
plt.plot(t,y_predict,'g-',linewidth=2,label='预测值')
plt.legend(loc='upper left')
plt.title("线性回归预测时间与功率之间的关系",fontsize=20)
plt.grid(b=True)
plt.show()
