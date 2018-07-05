# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/6/29
    Desc : 
'''

import os
import sys
import time


def makeTagDic(file_path):
    '''
    读取索引文件,存储到字典中
    :param file_path:
    :return:
    '''
    type_dict = {'spam': "1", "ham": "0"}
    index_file = open(file_path)
    index_dict = {}
    try:
        for line in index_file:
            arr = line.split(" ")
            if len(arr) == 2:
                key, value = arr
                value = value.replace("../data", "").replace("\n", "")
                index_dict[value] = type_dict[key.lower()]
    finally:
        index_file.close()
    return index_dict


def mailContentToDic(file_path):
    '''
    将邮件中的有用信息提取出来,存储到dic中
    :param file_path:
    :return:
    '''
    file = open(file_path, "r", encoding="gb2312", errors='ignore')
    content_dict = {}

    try:
        is_content = False  # 初始化为False后，在循环之外
        for line in file:
            line = line.strip()
            if line.startswith("From:"):
                # From: "yan"<(8月27-28,上海)培训课程>
                content_dict['from'] = line[5:]
            elif line.startswith("To:"):
                content_dict['to'] = line[3:]
            elif line.startswith("Date:"):
                content_dict['date'] = line[5:]
            elif not line:
                is_content = True

            # 处理邮件内容
            if is_content:
                if 'content' in content_dict:
                    content_dict['content'] += line
                else:
                    content_dict['content'] = line
    finally:
        file.close()
    return content_dict


def dicToText(file_path):
    '''
    处理邮件数据
    :param file_path:
    :return:
    '''
    content_dict = mailContentToDic(file_path)
    # 进行处理
    result_str = content_dict.get('from', 'unkown').replace(',', '').strip() + ","
    result_str += content_dict.get('to', 'unknown').replace(',', '').strip() + ","
    result_str += content_dict.get('date', 'unknown').replace(',', '').strip() + ","
    result_str += content_dict.get('content', 'unknown').replace(',', ' ').strip()
    return result_str


start = time.time()
index_dict = makeTagDic('./datas/full/index')
# for key, value in index_dict.items():
#     print(key, value)

list0 = os.listdir('./datas/data')
for l1 in list0:
    l1_path = './datas/data/' + l1
    print("开始处理文件夹 : ", l1_path)

    # 获取每个文件夹下的所有的文件列表
    list1 = os.listdir(l1_path)

    write_file_path = './datas/merge/process01_' + l1
    # 将每个文件夹下的所有的文件内容抽取出来后写入到一个文件中.其中一个文件一行
    with open(write_file_path, 'w', encoding='utf-8') as writer:
        for l2 in list1:
            l2_path = l1_path + "/" + l2
            index_key = "/" + l1 + "/" + l2

            if index_key in index_dict:
                content_str = dicToText(l2_path)
                content_str += "," + index_dict[index_key] + "\n"
                writer.writelines(content_str)

with open('./datas/result_process01', 'w', encoding='utf-8') as writer:
    for l1 in list0:
        file_path = './datas/merge/process01_' + l1
        print('开始合并文件: ', file_path)

        with open(file_path, encoding='utf-8') as file:
            for line in file:
                writer.writelines(line)

end = time.time()
print("数据处理总共耗时: ", (end - start))
