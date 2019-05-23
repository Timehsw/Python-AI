# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-16
    Desc : 
    Note : 
'''

from pyArango.connection import *
import os
import pandas as pd
import sys

from arango import ArangoClient

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)


clinet = ArangoClient(host="10.111.25.138")
db = clinet.db('GameOfThrones', username=":role:hushiwei", password="hushiwei")
mobile_phone=db.collection("MobilePhone")
print(mobile_phone.get('13804568133'))



a=db.collection("ChildOf")
print(a.get('192214'))