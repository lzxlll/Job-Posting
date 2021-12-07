# Job-Posting
The raw job posting data has a total number of observations of 140,107,287.

This project intends to:
1. Split the huge raw text data file into 142 files, with each file contains 1,000,000 online job postings (job_posting_subset.ipynb). Identify and drop the duplicated/missing value job vacancy postings (job_group_test.py).
2. Extract the 1000 most frequent job posting titles, then manually map to the standard Chinese occupation codes (job_group_test.py). 
3. Mapping the posting job titles to Chinese classification and codes of occupations (GB/T 6565-2015) (Continuous Bag of Words Model (CBOW)).
4. Assess the quality of the mapping by comparing mapping of the 1000 most frequent job postings by manual to the mapping by algorithm. 
5. Extract information from job descriptions. 

## ***Mapping Process***

#### 1. **Identify and drop the duplicated/missing value job vacancy postings**
We first delete the observations with missing job title information (obs = ?), this accounts for ?% of the total observations. We have the following informations:

| Variable | Format |
| --- | --- |
|招聘主键ID  | bigint(20)|
|公司ID  | bigint(20)|
|公司名称 | varchar(255)|
|城市名称 | varchar(255)|
|公司所在区域 | varchar(255)|
|工作薪酬 | varchar(255)|
|教育要求 | varchar(255)|
|工作经历 | varchar(255)|
|工作描述 | varchar(255)|
|职位名称 | varchar(255)|
|工作名称 | varchar(255)|
|招聘数量 | varchar(255)|
|发布日期  | datetime|
|行业名称 | varchar(255)|
|来源 | varchar(255)|

Second, we drop the duplicated job postings following the creteria: we sort the observations with same firm ID, firm location and job title in an ascending date order, the duplicated job postings refers to postings with identical firm ID and firm location and posted within the same year-month. This process drops ? which amounts to ?% of the total observations.

:tada: **Statistics**
- [ ] Number of observations dropped because of N/A in firm ID, firm location, posting date, job title and job description?
- [ ] Number of observations dropped because of duplicated job postings?
- [ ] What is the frequency distribution of the posting date, by year-day?
- [ ] What is the frequency distribution of the online job posting by source?


#### 2. **Extract the 1000 most frequent job posting titles**
To identify the 1000 most frequent job posting titles, we group the postings by job titles. Then, calculate the frequency within each group. There are 28,949,287 observations share the 1000 most frequent job posting titles, which amounts to ?% of the total observations.



#### 3. **Mapping the online posting titles to Chinese classification and codes of occupations (GB/T 6565-2015)**
The Chinese classification and codes of occupations (GB/T 6565-2015) [GB/T 6565-2015.xlsx](https://github.com/lzxlll/Job-Posting/files/7668463/default.xlsx) contains 4 levels of occupational classification: general (8 groups), medium (66 groups), detail (413 groups) and 1838 occupations. 

For each occupation, a job description and definition is attched. In particular, for each job title t in our online posting data, we compute the similarity between t and all of the job titles, τ, which appear in Chinese classification and codes of occupations (GB/T 6565-2015). For each standardized job title τ, we observe an GB/T occupation code. For the job title t, we assign to t the GB/T occupation code of the standardized job title τ. We do this for any job title that appears at least twice in our online job posting data. 



#### 4. **Mapping effectiveness check**
We rely on comparing the results of manual mapping and algorithm mapping to validate our practice. In `Extract the 1000 most frequent job posting titles` step we rank the online titles by frequency and select the top 1000 as candadats for manual mapping. We use human knowledge and solely rely on "job title" information to map top 1000 online job titles to Chinese classification and codes of occupations (GB/T 6565-2015) [our_chinese_mapping.xlsx](https://github.com/lzxlll/Job-Posting/files/7668450/our_chinese_mapping.xlsx)
.
 
This effectiveness check rely on two assumptions: (i) human-knowledge based manual mapping is the most precise one. (ii) online job posting's title and description should be matched. This is, job title "computer engineer" should has "computer engineer" related information in the job description rather than other arbitrary descriptions. 

:tada: **Statistics**

We compare this mapping result to the result based on algorithm in `Mapping the online posting titles to Chinese classification and codes of occupations (GB/T 6565-2015)` in the following ways:

- For each top 1000 online job titles, we check what percentage of algorithm based mapping's most likely mapping has the same mapping results as manual mapping. Be more specific, we have mentioned above that `28,949,287 observations share the 1000 most frequent job posting titles'. 

  Let's say there are 1,000 postings have online job title "sale agent", and we manually mapped it to "salesman" in the GB/T 6565-2015. On the other hand, by algorithm 30% of "sale agent" can be mapped to "teacher", 20% can be mapped to "cook", 50% can be mapped to "salesman". Then, in this case "salesman" is the most likely mapping and it matches with the manual mapping result. We calculate the percentage of this chance for all the top 1000 online job titles. 

- We compare the frequency distribution of 28,949,287 observations for manual based mapping and algorithm based mapping. This is, we compare the GB/T 6565-2015 title frequency distribution based on manual and algorithms. Ideally, both distributions should share similar patterns. 



<hr />

## ***Extract information Process***

Our task in this section is to characterize the text within each job description in terms of the skills, tasks, and other occupational elements described in the job ad.  Specifically, we count the apperance of the keywords in the description. By doing so, we decompose the texted description into a set of dummies variables. For instance, there are 67 job characteristics associated with techonology in our complied keywords file. Then, we created 67 columns with each one corresponding to a technology keywords. We count the apperance of each keywords and fill in the cell for each observation (job description). 

Our keywords build on "The Evolution of Work in the United States" by Enghin Atalay, Phai Phongthiengtham, Sebastian Sotelo and Daniel Tannenbaum - American Economic Journal: Applied Economics (2020) https://www.aeaweb.org/articles?id=10.1257/app.20190070. 

We first use google translator to convert each keyword (in English) to Chinese. Because keywords are related within each category, a direct translation results in multiple synonyms been dropped. To handle this issue, we rely on our CBOW model to expand the keywords list and manually screen for the CBOW model based key words. Therefore, our keywords list include (i) direct translation of keywords from "The Evolution of Work in the United States"; (ii) keywords generated from CBOW model. 

We extract information in the online job descriptions by searching for the key words. Our keywords file [our_chinese_mapping.xlsx](https://github.com/lzxlll/Job-Posting/files/7668240/our_chinese_mapping.xlsx) includes three spreadsheets (techonology, character, O*NET) with the same format: job characteristics, and our judgement. For each job character, we apply the CBOW to generate a list of CBOW keywords based on our judgement. 

- Usage of different technologies (e.h., Microsoft Word, Python, Matlab); 
  - reuqired programming technology, such as C++, python and so on. 
  - the usage of industrial system, such as SAP, 金碟 and so on.
  - the application of big data. 
  
- Character, financial skills, problem management skills and so on; Nonroutine (interactive, analytic, and manual) and routine (cognitive, and manual) tasks;
  - nonroutine (interactive, analytic, and manual) and routine (cognitive, and manual) tasks, which is build on Spitz-Oener (2006).
  - different skill-related words, which is build on Deming and Kahn (2017).
  - personality traits (Big 5), which is build on John, Naumann, and Soto (2008). 
  
- O*NET work styles, skills, knowledge requirements, and activities.




