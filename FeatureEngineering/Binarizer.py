# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/27
    Desc : 二值化
'''

'''
1. 一般情况下，对于每个特征需要使用不同的阈值进行操作，所以一般我们会拆分称为几个DataFrame进行二值化操作后，再将数据合并.
2. 一般情况下，对数据进行划分的时候，不是进行二值化，而是进行多值化(分区化/分桶化)；即：将一个连续的数据，按照不同的取值范围，分为不同的级别；比如：在某一个模型中，存在人的收入情况，单位为元，根据业务来判断的话，可能会得到影响因变量的因素其实是区间后的收入情况，那么这个时候就可以基于业务的特征，将收入划分为收入等级，比如：1w -> 0, 1w~2w -> 1, 2w~3w -> 2, 3w+ -> 3(不要对数据做哑编码，因为这里的0 1 2 3其实是有不同的相似度的/等级)
'''

import numpy as np
from sklearn.preprocessing import Binarizer

arr = np.array([
    [1.5, 2.3, 1.9],
    [0.5, 0.5, 1.6],
    [1.1, 2, 0.2]
])

# threshold就是阈值,大于这个阈值就是1,小于就是0
binarizer = Binarizer(threshold=2.0).fit(arr)
print(binarizer)
print(binarizer.transform(arr))
