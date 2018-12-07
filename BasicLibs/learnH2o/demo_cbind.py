# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/12/7
    Desc : 
    Note : 
'''

import h2o
h2o.init()
import numpy as np

# Generate a random dataset with 10 rows 4 columns.
# Label the columns A, B, C, and D.
cols1_df = h2o.H2OFrame.from_python(np.random.randn(10,4).tolist(), column_names=list('ABCD'))

print(cols1_df.describe)

# Generate a second random dataset with 10 rows and 1 column.
# Label the columns, Y and D.
cols2_df = h2o.H2OFrame.from_python(np.random.randn(10,2).tolist(), column_names=list('YZ'))
print(cols2_df.describe)

# Add the columns from the second dataset into the first.
# H2O will append these as the right-most columns.
colsCombine_df = cols1_df.cbind(cols2_df)
print(colsCombine_df.describe)