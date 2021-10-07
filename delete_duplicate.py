import os
import pandas as pd
import csv
path_0 = '/share/home/320346/job_posting'
path_1 = '/share/home/320346/base_data'
filelist = os.listdir(path_0)  # 目录下文件列表

for files in filelist:

    dir_path = os.path.join(path_0, files)
    # 分离文件名和文件类型
    file_name = os.path.splitext(files)[0]  # 文件名
    file_type = os.path.splitext(files)[1]  # 文件类型

    # 将.dat文件转为.csv文件
    if file_type == '.dat':  # 可切换为.xls等
        file_test = open(dir_path, 'rb')
        new_dir = os.path.join(path_1, str(file_name) + '.csv')
        file_test2 = open(new_dir, 'wb')  # 创建/修改新文件
        for lines in file_test.readlines():
            lines=lines.decode()
            lines = lines.replace(',', '，')
            str_data = ",".join(lines.split('@!'))  # 分隔符
            file_test2.write(str_data.encode('utf-8-sig'))
        file_test.close()
        file_test2.close()
        data = pd.read_csv(new_dir,quoting=csv.QUOTE_NONE, header=None, names=['招聘主键ID', '公司ID', '公司名称', '城市名称', '公司所在区域', '工作薪酬', '教育要求', '工作经历', '工作描述', '职位名称', '工作名称', '招聘数量', '发布日起', '行业名称', '招聘平台'])  # 将.csv文件转换为dataframe数据
       # data['发布日起'] = pd.to_datetime(data['发布日起'], format="%Y-%m-%d")

        def get_date(x):
            df=x.sort_values(by = '发布日起',ascending=True)
            return df.iloc[-1, :]

        # 删除重复数据
        if any(data.duplicated(subset=['公司ID', '工作名称'])):
            data_1=data.groupby(['公司ID','工作名称'],as_index=False).apply(get_date)
            data_1.to_csv('/share/home/320346/delete_data//'+ str(file_name)+ '.csv',encoding='utf-8-sig')


















