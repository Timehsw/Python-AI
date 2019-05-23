# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019-04-17
    Desc : 
    Note : 
'''
import os
import pandas as pd
import numpy as np

root_path = "/Users/hushiwei/Downloads/模拟数据"
os.chdir(root_path)

apply_path = "apply.csv"
apply_df = pd.read_csv(apply_path, error_bad_lines=False)

print(apply_df)

# 排除中文中以数字开头的数字部分

apply_df['workName'] = apply_df['workName'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)
apply_df['workAddr'] = apply_df['workAddr'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)
apply_df['contactAddr'] = apply_df['contactAddr'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)
apply_df['homeAddr'] = apply_df['homeAddr'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)
apply_df['idAddr'] = apply_df['idAddr'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)
apply_df['houseAddr'] = apply_df['houseAddr'].str.extract(u"([^0-9][\u4e00-\u9fa5].+)", expand=False)

tmp = apply_df['phone'].str.extract(u"^(0\d{2,3}-\d{7,8})|(1[356784]\d{9})$", expand=False)  # 中国座机
apply_df['phone'] = tmp.apply(lambda line: line[0] if line[0] is not np.NaN else line[1], axis=1)

apply_df['mobilePhone'] = apply_df['mobilePhone'].astype(str)
apply_df['mobilePhone'] = apply_df['mobilePhone'].str.extract(u"^(1[356784]\d{9})$", expand=False) # 手机号

apply_df['relationMobile'] = apply_df['relationMobile'].astype(str)
apply_df['relationMobile'] = apply_df['relationMobile'].str.extract(u"^(1[356784]\d{9})$", expand=False) # 手机号

apply_df.fillna("Empty", inplace=True)

apply_df.to_csv("./apply_nomissing.csv", index=False)
