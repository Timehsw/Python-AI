# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from pyArango.connection import *
import os
import pandas as pd

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)
# customid=apply_df["customNum"].values.tolist()
conn = Connection(arangoURL='http://10.111.25.138:8529', username=":role:hushiwei", password="hushiwei", )

# db=conn.createDatabase(name="GameOfThrones")
db = conn['GameOfThrones']
print(db)

customidCollection = db.createCollection(name="CustomIds")


for customid in apply_df["customNum"]:
    doc = customidCollection.createDocument()
    doc["name"] = customid
    doc._key = customid
    doc.save()
