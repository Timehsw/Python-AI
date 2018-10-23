#!/usr/bin/env python 

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as image
import base64
from sklearn.externals.six import StringIO
import pydotplus

from io import BytesIO
from IPython.display import Image

data = pd.read_csv("./datas/xunlian.csv")
train_data = data.drop("GEO_Y", axis=1)
target = data["GEO_Y"]

DTmodel = DecisionTreeClassifier(max_depth=4)
DTmodel = DTmodel.fit(train_data, target)
print(DTmodel)

# 将决策树的图片转成base64
dot_data = tree.export_graphviz(DTmodel, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)

pic_base64 = base64.b64encode(Image(graph.create_png()).data).decode('utf8')

print(pic_base64)
