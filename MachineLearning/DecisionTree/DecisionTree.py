#!/usr/bin/env python 

import base64

import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./datas/xunlian.csv")
train_data = data.drop("GEO_Y", axis=1)
target = data["GEO_Y"]

DTmodel = DecisionTreeClassifier(max_depth=4)
DTmodel = DTmodel.fit(train_data, target)
print(DTmodel)

# 将决策树的图片转成base64
dot_data = tree.export_graphviz(DTmodel, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
pic_base64_ut8 = base64.b64encode(Image(graph.create_png()).data).decode('utf8')
pic_base64 = base64.b64encode(Image(graph.create_png()).data)

print(pic_base64_ut8)
print(pic_base64)
print(graph.create_png())