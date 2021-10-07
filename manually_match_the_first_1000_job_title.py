
import numpy as np
from gensim import corpora, models, similarities
from gensim.models import Word2Vec, keyedvectors
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary
import jieba.posseg as jp, jieba
import pandas as pd
from string import punctuation
import re
import sys


jieba.enable_paddle()

add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc
model_path = '/share/home/320346/baike_26g_news_13g_novel_229g.model'
job_path = '/share/home/320346/occupation_cat_china.xlsx'
job_post_path = '/share/home/320346/extract_data/排序_1.csv'


model = Word2Vec.load(model_path)
word_vectors = model.wv


def jieba_cut(sentence, word_vectors):
    word_list = []
    seg_list = jieba.cut_for_search(sentence)
    for word in seg_list:
        if word != '' and word is not None and word not in all_punc and word in word_vectors.key_to_index:
            word_list.append(word)
    return word_list


def keep_all_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def create_job_categories_map(job_dictionary_file):
    job_categories_dic = dict()
    job_categories_describ_list_dic = dict()
    for index, row in job_dictionary_file.iterrows():
        job_id = row['职业代码']
        if row['定义'] and not row['定义'].isspace():
            job_describ = row['定义'] + row['职责'] + row['大类'] + row['中类'] + row['小类'] + row['细类']
            job_categories_dic[job_id] = row['大类'] + row['中类'] + row['小类'] + row['细类']
            job_describ_word_list = list(set(jieba_cut(keep_all_chinese(job_describ), word_vectors)))
            job_categories_describ_list_dic[job_id] = job_describ_word_list

    return job_categories_dic, job_categories_describ_list_dic

def find_job_category(job_name, job_categories_describ_list_dic, job_categories_dic):
    job_name_list = []
    seg_list = jieba_cut(job_name, word_vectors)
    for word in seg_list:
        if word != '' and word is not None and word not in all_punc and word in word_vectors.key_to_index:
            job_name_list.append(word)
    # 1st compare with descriptio
    job_scores = dict()
    for key in job_categories_describ_list_dic:
        job_scores[key] = word_vectors.n_similarity(job_categories_describ_list_dic[key], job_name_list)

    job_scores = sorted(job_scores.items(), key=lambda d: d[1], reverse=True)
    # 2nd compare with title
    job_scores_des = dict()
    for index in range(0, 5):
        category_1 = job_scores[index][0]
        job_category_name = list(set(jieba_cut(keep_all_chinese(job_categories_dic[category_1]), word_vectors)))
        job_scores_des[category_1] = word_vectors.n_similarity(job_category_name, job_name_list)
    # Return top job
    return sorted(job_scores_des.items(), key=lambda d: d[1], reverse=True)



data = pd.read_excel(job_path, keep_default_na=False)
job_dictionary_file = pd.DataFrame(data, columns=['职业代码', '大类', '中类', '小类', '细类', '职责', '定义'])
job_categories_dic, job_categories_describ_list_dic = create_job_categories_map(job_dictionary_file)

posted_jobs_data = pd.read_csv(job_post_path, header=None, names=['工作名称', '数量'],encoding='gbk')
posted_jobs = pd.DataFrame(posted_jobs_data, columns=['工作名称']).values.tolist()



for post_job in posted_jobs:
    post_job_descri_chinese = keep_all_chinese(post_job[0])
    if len(post_job_descri_chinese) != 0:
        job_scores_des = find_job_category(post_job_descri_chinese, job_categories_describ_list_dic, job_categories_dic)
        mapping_result = pd.DataFrame([[post_job[0], job_categories_dic[job_scores_des[0][0]], job_scores_des[0]]],
                                      columns=['post_job', 'category', 'sores'])
        mapping_result.to_csv('/share/home/320346/first_post_mapping/the_first_1000_job_title_mapping.csv', header=0, mode='a',
                              encoding='utf-8-sig', index=False)





