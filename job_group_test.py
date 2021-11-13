# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import gc, json, csv, re

def read_csv_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源']
    resCSV = pd.read_csv(curFile, header=None, index_col=None, names=colNames,encoding='utf-8',quoting=csv.QUOTE_NONE)
    #parse_dates=['date'])
    return resCSV

def read_dat_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源']
    resCSV = pd.read_csv(curFile, header=None, index_col=None, names=colNames,encoding='utf-8',quoting=csv.QUOTE_NONE, sep="@!", error_bad_lines=False, engine='python')
    return resCSV


dataNameTmp = "job_posting/job_posting_%s.dat"
csvOutTmp = "csv_out/job_res_%s.csv"
json_filename = 'csv_out/datetag.json'
dateJson = {}

resColList = ['job_title','title_nums']
totalDf = pd.DataFrame(columns=resColList)

naSum = 0
dupSum = 0

for i in range(1,142):
    curFile = dataNameTmp%i
    print(curFile," is Grouping Computing...")
    datDf = read_dat_data(curFile)
    #datDf.loc[datDf['职位名称'].str.contains('%',na=False),'职位名称'] = np.NaN
    
    datDf = datDf.replace(r'\N',np.NaN).dropna(subset=['工作名称','发布日期'])
    datDf.loc[datDf['职位名称'].isna(),'职位名称'] = datDf.loc[datDf['职位名称'].isna(),'工作名称']
    datDf.loc[datDf['职位名称'].isin(["兼职","全职"]),'职位名称'] = datDf.loc[datDf['职位名称'].isin(["兼职","全职"]),'工作名称']
    
    curNa = datDf.shape[0]
    naSum = naSum + curNa
    
    datDf['date'] = datDf['发布日期'].apply(lambda x: x[0:7])
    datDf = datDf.drop_duplicates(subset=['公司ID','职位名称','公司所在区域','date'], keep='first').reset_index(drop=True)
    
    curDup = datDf.shape[0]
    dupSum = dupSum + curDup
    
    print(curFile,"删除空值剩余: %s"%curNa, "去重复值剩余：%s"%curDup)
    
    #resDf = datDf.groupby(['职位名称']).count()[['工作名称']]
    resDf = datDf.groupby(['工作名称']).count()[['职位名称']]
    resDf = resDf.reset_index(drop=False)
    resDf.columns = resColList
    totalDf = totalDf.append(resDf,ignore_index=True)
    
    del datDf
    gc.collect()
    
#totalDf.to_excel("init_groupby.xlsx")
#totalDf.to_csv("init_groupby.txt",index=False)
#totalDf.loc[0:1000].to_csv("init_groupby_1000.txt",sep='?',index=False)
totalDf['title_nums'] = pd.to_numeric(totalDf['title_nums'], errors='coerce')
totalDf['job_title'] = totalDf['job_title'].apply(lambda x: (re.sub(r'兼职|全职','',x)))
totalDf['job_title'] = totalDf['job_title'].apply(lambda x: (re.sub(r'\?',' ',x)))
gpSumDf = totalDf.groupby(['job_title']).sum().reset_index()

print(totalDf.groupby(['job_title']).sum().head(10))
print("------------------------------")
print(gpSumDf.head(10))
print("------------------------------")
print(totalDf.head(10))

gpSumDf = gpSumDf.sort_values(by="title_nums",axis=0,ascending=False).reset_index(drop=True)

gpSumDf.loc[0:1000].to_excel("gp_res_1000.xlsx", index=False)
gpSumDf.to_csv("init_groupby.csv",sep='?', encoding = 'utf_8_sig', index=False)

#print("naSum:",naSum)
#print("dupSum:",dupSum)
