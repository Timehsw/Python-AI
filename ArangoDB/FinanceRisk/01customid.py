# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 获取客户编号
    Note : 将客户编号入库
'''

import os
import pandas as pd
import sys

from arango import ArangoClient

clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")
if db.has_collection("CustomIds"):
    CustomIdsCol = db.collection("CustomIds")
else:
    CustomIdsCol = db.create_collection("CustomIds")

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply_nomissing.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

# 获取客户编号
result = [{'_key': customid, 'name': customid} for customid in apply_df["customNum"]]

CustomIdsCol.insert_many(result)
