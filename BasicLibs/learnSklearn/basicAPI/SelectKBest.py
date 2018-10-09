# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/10/8
    Desc : 卡方检验
    Note : 
'''
import numpy as np
import pandas as pd
from statsmodels.datasets import longley
import statsmodels.formula.api as smf

nobs = 100
np.random.seed(987125)
yx = np.random.randn(nobs, 2)
beta0 = 0
beta1 = 1
yx[:, 0] += beta0 + beta1 * yx[:, 1]
data = pd.DataFrame(yx, columns=['TOTEMP', 'GNP'])

hypothesis_0 = '(Intercept = 0, GNP = 0)'
hypothesis_1 = '(GNP = 0)'
hypothesis_2 = '(GNP = 1)'
hypothesis_3 = '(Intercept = 0, GNP = 1)'
results = smf.ols('TOTEMP ~ GNP', data).fit()
wald_0 = results.wald_test(hypothesis_0)
wald_1 = results.wald_test(hypothesis_1)
wald_2 = results.wald_test(hypothesis_2)
wald_3 = results.wald_test(hypothesis_3)

print('H0:', hypothesis_0)
print(wald_0)
print()
print('H0:', hypothesis_1)
print(wald_1)
print()
print('H0:', hypothesis_2)
print(wald_2)
print()
print('H0:', hypothesis_3)
print(wald_3)