# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2020/4/1
    Desc : 批量递归修改文件名
    Note : 
'''

import os
def file_name_walk(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print("root", root)  # 当前目录路径
        # print("dirs", dirs)  # 当前路径下所有子目录
        # print("files", files)  # 当前路径下所有非目录子文件

        if len(dirs)==0:
            # print(root)
            # print(files)
            for file_name in files:
                # print(file_name)
                new_file_name=file_name.replace("获取更多资源公众号：哆啦A梦宝藏库","").replace("获取更多资源关注：哆啦A梦宝藏库","")
                os.renames("{}/{}".format(root,file_name),"{}/{}".format(root,new_file_name))
                print("{}/{}".format(root,file_name),"{}/{}".format(root,new_file_name))

filePath="E:/BaiduNetdiskDownload/He Ji"

file_name_walk(filePath)