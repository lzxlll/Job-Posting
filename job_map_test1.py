# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc, json, csv, re, time
from string import punctuation
from collections import Counter

from time import time


from gensim import corpora, models, similarities
from gensim.models import Word2Vec,keyedvectors 
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import threading
import timeit

import multiprocessing as mp
from concurrent.futures import as_completed,wait
import datetime

import jieba
import jieba.analyse
import jieba.posseg as pseg
from jieba import analyse

tfidf = analyse.extract_tags
textrank = analyse.textrank
analyse.set_stop_words('baidu_stopwords.txt') 

jieba.enable_paddle()
#jieba.enable_parallel(4)
jieba_paddle_exlist = ['f','t','vd','a','ad','d','m','q','r','p','c','u','xc','w','TIME','PER']
jieba_inlist = ['n','s','nr','ns','nt','nz','nw','v','vn','an','ORG','LOC']



def read_csv_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源']
    resCSV = pd.read_csv(curFile, header=None, index_col=None, names=colNames,encoding='utf-8',quoting=csv.QUOTE_NONE,error_bad_lines=False)
    #parse_dates=['date'])
    return resCSV

def read_dat_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源']
    resCSV = pd.read_csv(curFile, header=None, index_col=None, names=colNames,encoding='utf-8',quoting=csv.QUOTE_NONE, sep="@!", error_bad_lines=False, engine='python')
    return resCSV


def read_txtcsv_data(curFile):
    colNames = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源']
    resCSV = pd.read_csv(curFile, header=None, index_col=None, names=colNames,encoding='utf-8', quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='?', engine='python')
    return resCSV


def getMaxIndex(sDf,cText,cWv):
    sRes = sDf['desc'].apply(lambda x: 0 if len(cText)==0 or len(x)==0 else cWv.n_similarity(cText,x))
    return sRes.idxmax()

def getStopWords():
    with open('baidu_stopwords.txt', encoding='utf8') as file:
        line_list = file.readlines()
        stopword_list = [k.strip() for k in line_list]
        stopword_set = set(stopword_list)
        print('停顿词列表，即变量stopword_list中共有%d个元素' %len(stopword_list))
        print('停顿词集合，即变量stopword_set中共有%d个元素' %len(stopword_set))
    return stopword_list


def transTextArr2Mat3(textArr,stopwords,wv=None):
    retexts = []
    for i,sentence in enumerate(textArr):
        cur_senWords = []
        sentence = str(sentence)
        sentence = re.sub('[^\u4e00-\u9fa5]+','',sentence)
        if len(sentence) > 0:
            for word,flag in pseg.cut(sentence):
                if  word not in stopwords and flag not in jieba_paddle_exlist and len(word)>1:
                    if wv is not None:
                        if word in wv.key_to_index:
                            cur_senWords.append(word)
                        else:
                            continue
                    else:
                        cur_senWords.append(word)
            cur_senWords = list(set(cur_senWords))
            retexts.append(cur_senWords)
        else:
            retexts.append(cur_senWords)
    return np.array(retexts,dtype=object)

def transTextArr2Mat31(textArr,stopwords,wv=None):
    retexts = []
    for i,sentence in enumerate(textArr):
        cur_senWords = []
        sentence = str(sentence)
        sentence = re.sub('[^\u4e00-\u9fa5]+','',sentence)
        if len(sentence) > 0:
            for word in jieba.lcut(sentence):
                if  word not in stopwords and len(word)>1:
                    if wv is not None:
                        if word in wv.key_to_index:
                            cur_senWords.append(word)
                        else:
                            continue
                    else:
                        cur_senWords.append(word)
            cur_senWords = list(set(cur_senWords))
            retexts.append(' '.join(cur_senWords))
        else:
            retexts.append(' '.join(cur_senWords))
        
    return np.array(retexts,dtype=object)

def transTextArr2Mat32(textArr,stopwords,wv=None):
    retexts = []
    for i,sentence in enumerate(textArr):
        cur_senWords = []
        sentence = str(sentence)
        sentence = re.sub('[^\u4e00-\u9fa5]+','',sentence)
        if len(sentence) > 0:
            for word in jieba.lcut(sentence):
                if  word not in stopwords and len(word)>1:
                    if wv is not None:
                        if word in wv.key_to_index:
                            cur_senWords.append(word)
                        else:
                            continue
                    else:
                        cur_senWords.append(word)
            cur_senWords = list(set(cur_senWords))
            retexts.append(cur_senWords)
        else:
            retexts.append(cur_senWords)
    return np.array(retexts,dtype=object)

def run_time(func):  
    def wrapper(*args, **kw):  
        start = timeit.default_timer()
        func(*args, **kw) 
        end = timeit.default_timer()
        runTime = round((end - start),2)
        print("runtime: ",runTime)
    return wrapper


def pTest(st,ed):
    print(f"{st}-{ed} \n")
    transDf = None
    s0 = time()       
    matDf = transTextArr2Mat3((datDf['职位名称']+datDf['工作名称']).iloc[st:ed].values, g_stopwords, word_vectors)
    transDf = stDf.loc[pd.Series(matDf).apply(lambda x: getMaxIndex(stDf,x,word_vectors))].reset_index(drop=True)
    transDf = pd.concat([datDf.iloc[st:ed,:].reset_index(drop=True),transDf.iloc[:,3:7]],axis=1)
    resStr = f"runtime: {time()-s0},shape:{transDf.shape}"
    print(resStr)
    return transDf


g_stopwords = getStopWords()
add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc

dataNameTmp = "map_first/job_res_%s.csv"
csvOutTmp = "map_res/job_res_%s.csv"
json_filename = 'map_res/datetag.json'


model = Word2Vec.load("job_wv.model")
word_vectors = model.wv


stDf = pd.read_excel('occupation_cat_china.xlsx', index_col=None)
stDf = stDf.dropna(subset=['细类']).reset_index(drop=True).fillna('')
trainSens = transTextArr2Mat32((stDf['小类'] + stDf['细类'] + stDf['定义'] + stDf['职责']).values, g_stopwords,word_vectors)
stDf['desc'] = trainSens


def mpool(tsize,tOut):
    maxProcs = 10
    nsize = 100000
    procs = int(tsize/nsize) + 1
    stm = time()
    
    resColList = ['招聘ID','公司ID','公司名称','城市名称','公司所在区域','工作薪酬','教育要求','工作经历',
                '工作描述','职位名称','工作名称','招聘数量','发布日期','行业名称','数据来源','大类','中类','小类','细类']
    tDf = pd.DataFrame(columns=resColList)
    
    with ProcessPoolExecutor(max_workers=maxProcs) as tpe:
        taskList = []
        for i in range(0,procs):
            sti = i*nsize
            edi = (i+1)*nsize if (i+1)*nsize < tsize else tsize
            obj = tpe.submit(pTest, sti, edi)
            taskList.append(obj)
        for taskItem in as_completed(taskList):
            resDf = taskItem.result()
            tDf = tDf.append(resDf,ignore_index=True)
            #print(result)
        #wait(taskList)
    tDf.to_csv(tOut,sep='?', encoding = 'utf_8_sig', index=False, header=False)
    del tDf
    print('total run time: %.3f s'%(time()-stm))
    

if __name__ == '__main__':
    
    
    for i in range(1,142):
        curFile = dataNameTmp%i
        curOut = csvOutTmp%i
        
        datDf = read_txtcsv_data(curFile)
        totalSize = datDf.shape[0]
        print(curFile," is Mapping...")
        print("------------------------------")
        mpool(totalSize,curOut)
        del datDf
        gc.collect()
        
    

