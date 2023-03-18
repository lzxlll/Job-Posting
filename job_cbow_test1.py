# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


import gc, json, csv, re

from string import punctuation
from collections import Counter
from time import time

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import threading
import timeit

import multiprocessing as mp
from concurrent.futures import as_completed,wait

import jieba
import jieba.analyse
import jieba.posseg as pseg

#jieba.enable_paddle()


import warnings
warnings.filterwarnings("ignore")



def transText2Arr(sentence):
    retexts = []
    cur_senWords = []
    sentence = re.sub('[^\u4e00-\u9fa5]+','',sentence)
    if len(sentence) > 0:
        for word,flag in pseg.cut(sentence):
            cur_senWords.append(word)
        cur_senWords = list(set(cur_senWords))
        retexts.extend(cur_senWords)
    else:
        retexts.extend(cur_senWords)
    return retexts

def read_restxtcsv_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源','大类','中类','小类','细类']
    resCSV = pd.read_csv(curFile, header=None,index_col=None, names=colNames,encoding='utf-8', quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='?', engine='python')
    return resCSV

def cbowMatch(x,xitem):
    icount = 0
    judgeList = istDf.loc[xitem]['cbow']
    for listItem in x:
        if listItem in judgeList:
            icount = icount + 1
    return icount

def judgeMatch(x,xitem):
    icount = 0
    judgeList = istDf.loc[xitem]['judge']
    for listItem in x:
        if listItem in judgeList:
            icount = icount + 1
    return icount

def pTest(st,ed):
    print(f"{st}-{ed} \n")
    transDf = None
    s0 = time()
    
    curResDf = resDf.iloc[st:ed]
    
    curResDf = curResDf[~curResDf['工作描述'].astype(str).str.contains('None')].reset_index(drop=True)   
    curResDf['工作描述'] = curResDf['工作描述'].apply(lambda x: transText2Arr(str(x)))
    curResDf = curResDf[['招聘ID','工作描述']]
    
    curResDf[stDf['vid'] + '_judge'] = np.NaN
    curResDf[stDf['vid'] + '_cbow'] = np.NaN
    
    destColList = stDf['vid'].values.tolist()
    for colItem in destColList:
        curResDf[colItem + '_judge'] = curResDf['工作描述'].apply(lambda x: judgeMatch(x,colItem))
        curResDf[colItem + '_cbow'] = curResDf['工作描述'].apply(lambda x: cbowMatch(x,colItem))
    
    curResDf.drop(columns=['工作描述'],inplace=True)
    resStr = f"runtime: {time()-s0},shape:{curResDf.shape}"
    print(resStr)
    return curResDf

def mpool(tsize,tOut):
    maxProcs = 10
    nsize = 100000
    procs = int(tsize/nsize) + 1
    stm = time()
    
    colNames = ['招聘ID']
    colNames.extend((stDf['vid'] + '_judge').values.tolist())
    colNames.extend((stDf['vid'] + '_cbow').values.tolist())
    
    tDf = pd.DataFrame(columns=colNames)
    
    with ProcessPoolExecutor(max_workers=maxProcs) as tpe:
        taskList = []
        for i in range(0,procs):
            sti = i*nsize
            edi = (i+1)*nsize if (i+1)*nsize < tsize else tsize
            obj = tpe.submit(pTest, sti, edi)
            taskList.append(obj)
        for taskItem in as_completed(taskList):
            retDf = taskItem.result()
            tDf = tDf.append(retDf,ignore_index=True)
            
    tDf.to_csv(tOut,sep='?', encoding = 'utf_8_sig', index=False, header=False)
    del tDf
    print('total run time: %.3f s'%(time()-stm))
    

stDf = pd.read_excel('our_chinese_mapping.xlsx', index_col=None)
stDf['judge'] = stDf['judge'].apply(lambda x: [item.strip() for item in x.split(',')])
stDf['cbow'] = stDf['cbow'].fillna("")
stDf['cbow'] = stDf['cbow'].apply(lambda x: [item.strip() for item in x.split(',')])

istDf = stDf.set_index('vid')

dataNameTmp = "mapped_job_posting/job_res_%s.csv"
cbowOutTmp = "cbow_out_res/job_res_%s.csv"

for i in range(1,2):
    
    curFile = dataNameTmp%i
    curCbowOut = cbowOutTmp%i
    print("%s is cbow..."%curFile)
    try:
        resDf = read_restxtcsv_data(curFile)
    except:
        continue
    totalSize = resDf.shape[0]
    mpool(totalSize,curCbowOut)
    del resDf
    gc.collect()
    print('%s run time: %.3f s'%(curFile,(time()-stm)))
    
#totalDf.to_excel("restat_res.xlsx",encoding='utf-8')