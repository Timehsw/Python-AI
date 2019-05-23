# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-17
    Desc : 
    Note : 
'''

import json
import os

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

# read
with open("./homeAddr_id.json", "r") as fp:
    workname_id=json.load(fp)

workname_id
'南京市高淳县东坝镇青山路58号'