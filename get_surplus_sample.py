import os
import pandas as pd

path = '/share/home/320346/delete_data'
filelist = os.listdir(path)  # 目录下文件列表

# 删除重复数据中前1000个工作名称的样本
for files in filelist:

    dir_path = os.path.join(path, files)
    # 分离文件名和文件类型
    file_name = os.path.splitext(files)[0]  # 文件名
    file_type = os.path.splitext(files)[1]  # 文件类型

    new_dir = os.path.join(path, str(file_name) + '.csv')

    sample = pd.read_csv(new_dir)
    name = pd.read_csv('/share/home/320346/extract_data/排序.csv', header=None, names=['工作名称', '数量'])

    list_1 = name['工作名称'][-1001:].tolist()
    delete_sample = sample[~sample['工作名称'].isin(list_1)]
    delete_sample.to_csv('/share/home/320346/surplus_data//' + str(file_name) + '.csv', encoding='utf-8-sig')  # 将其保存到不同csv文件下。



