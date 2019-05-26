# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 17:52
   desc: the project
"""
from urllib import request

urls_file = './urls.txt'  # 保存的url链接
path = 'data/images/train/'  # 图片保存的文件夹

file = open(urls_file, 'r')

i = 745

for line in file:
    try:
        iamge_path = path + str(i) + '.jpg'
        request.urlretrieve(line, iamge_path)
        print(i)
        i = i + 1
    except:
        print("time out")

file.close()
