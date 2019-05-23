# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018-12-25
    Desc : 
    Note : 
'''

import os

fileName = "./title.txt"
import io

with io.open(fileName, "r", encoding="utf-8") as my_file:
    my_unicode_string = my_file.read()

lines = my_unicode_string.split("\n")

dic = {}
for index, line in enumerate(lines):
    # print(index + 1, line)
    dic[index + 1] = line+".mp4"

path="/Users/hushiwei/OneDrive/编程视频/数据分析/秦路王牌好课 七周成为数据分析师"
# path = "/Users/hushiwei/Downloads/秦路王牌好课 七周成为数据分析师"
os.chdir(path)
files = []
# r=root, d=directories, f = files
import re
for r, d, f in os.walk(path):
    for file in f:
        if '.mp4' in file:
            files.append(file)
            os.rename(file,dic[int(re.findall(r"\d+\.?\d*",file)[0])])
            print(file,'-',dic[int(re.findall(r"\d+\.?\d*",file)[0])])
            # files.append(os.path.join(r, file))

# for f in files:
#     print(f)

print(len(files))
print(len(lines))
