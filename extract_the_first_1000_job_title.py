import os
import pandas as pd

path = '/share/home/320346/delete_data'

filelist = os.listdir(path)  # 目录下文件列表

for files in filelist:

    dir_path = os.path.join(path, files)
    # 分离文件名和文件类型
    file_name = os.path.splitext(files)[0]  # 文件名
    file_type = os.path.splitext(files)[1]  # 文件类型

    new_dir = os.path.join(path, str(file_name) + '.csv')
    data_1 = pd.read_csv(new_dir)  # 将.csv文件转换为dataframe数据

    # 提取每个文件前1000个频率最高的

    data_2 = data_1.groupby('工作名称')
    a = data_2['工作名称'].count().sort_values()
    a[-1000:].to_csv('/share/home/320346/extract_data/数据.csv', header=0, mode='a', encoding='utf-8-sig')

# 提取所有数据文件中的前1000个

data_sort = pd.read_csv('/share/home/320346/extract_data/数据.csv', header=None, names=['工作名称', '数量'])
data_sort_1 =data_sort.groupby('工作名称')['数量'].sum().sort_values()
data_sort_1[-1100:].to_csv('/share/home/320346/extract_data/排序.csv', header=0, encoding='utf-8-sig')
