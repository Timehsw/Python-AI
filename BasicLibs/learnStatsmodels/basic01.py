# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/8
    Desc : 
    Note : 
'''
# 调用statsmodels里面的api，通过api调用相当于调用了statsmodels.regression.linear_model，可以使用linear_model文件里的函数
import statsmodels.api as sm

spector_data = sm.datasets.spector.load()  # 读取样例的数据集
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
# Fit and summarize OLS model
mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())
