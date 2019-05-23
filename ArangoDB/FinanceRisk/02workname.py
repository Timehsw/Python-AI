# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 获取公司名称
    Note : 将公司名称入库
'''

import os
import pandas as pd
import sys

from arango import ArangoClient

clinet = ArangoClient(host="10.111.32.65")
# db = clinet.db('financerisk', username="root", password="hxixpLi")
db = clinet.db('financerisk', username="hushiwei", password="hushiwei")
print(db.collections())
if db.has_collection("WorkNames"):
    work_nameCol = db.collection("WorkNames")
else:
    work_nameCol = db.create_collection("WorkNames")

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply_nomissing.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

# 获取公司名称,去除中文中的数字
# apply_df['workName'] = apply_df['workName'].str.extract(u"([\u4e00-\u9fa5]+)", expand=False)
workname_df = apply_df['workName']
workname_df.drop_duplicates(inplace=True)

result = [{'_key': str(index), 'name': work_name} for index, work_name in workname_df.iteritems()]
workname_id = {work_name: str(index) for index, work_name in workname_df.iteritems()}
import json

# write
with open("./workname_id.json", "w") as fp:
    json.dump(workname_id, fp)
# read
# with open("./workname_id.json", "r") as fp:
#     workname_id=json.load(fp)

work_nameCol.insert_many(result)
