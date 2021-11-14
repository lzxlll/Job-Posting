# Job-Posting
The total number of observations we acquired from raw job posting data is 140,107,287.

This project intends to:
1. identify and drop the duplicated/missing value job vacancy postings (job_group_test.py).
2. extract the 1000 most frequent job posting titles, then manually map to the standard Chinese occupation codes (job_group_test.py). 
3. mapping the posting job titles to Chinese classification and codes of occupations (GB/T 6565-2015) (Continuous Bag of Words Model (CBOW)).
4. assess the quality of the mapping by comparing mapping of the 1000 most frequent job postings by manual to the mapping by algorithm. 
5. extract information from job descriptions. 

## Mapping Process

### Identify and drop the duplicated/missing value job vacancy postings
We first delete the observations with missing job title information (obs = ?), this accounts for ?% of the total observations. 
Second, we drop the duplicated job postings following the creteria: we sort the observations with same firm ID, firm location and job title in an ascending date order, the duplicated job postings refers to postings with identical firm ID and firm location and posted within the same year-month. This process drops ? which amounts to ?% of the total observations.

- Statistics: 
  - 





### Extract the 1000 most frequent job posting titles
To identify the 1000 most frequent job posting titles, we group the postings by job titles. Then, calculate the frequency within each group. There are 28,949,287 observations share the 1000 most frequent job posting titles, which amounts to ?% of the total observations.


### Mapping the online posting titles to Chinese classification and codes of occupations (GB/T 6565-2015)
The Chinese classification and codes of occupations (GB/T 6565-2015) contains 4 levels of occupational classification: general (8 groups), medium (66 groups), detail (413 groups) and 1838 occupations. 

For each occupation, a job description and definition is attched. In particular, for each job title t in our online posting data, we compute the similarity between t and all of the job titles, τ, which appear in Chinese classification and codes of occupations (GB/T 6565-2015). For each standardized job title τ, we observe an GB/T occupation code. For the job title t, we assign to t the GB/T occupation code of the standardized job title τ. We do this for any job title that appears at least twice in our online job posting data. 

### Mapping effectiveness check
We rely on comparing the results of manual mapping and algorithm mapping to validate our practice. In `Extract the 1000 most frequent job posting titles` step we rank the online titles by frequency and select the top 1000 as candadats for manual mapping. We use human knowledge and solely rely on ``job title" information to map top 1000 online job titles to Chinese classification and codes of occupations (GB/T 6565-2015). This effectiveness check rely on two assumptions: (i) human-knowledge based manual mapping is the most precise one. (ii) online job posting's title and description should be matched. This is, job title `computer engineer' should has `computer engineer' related information in the job description rather than other arbitrary descriptions. 

We compare this mapping result to the result based on algorithm in `Mapping the online posting titles to Chinese classification and codes of occupations (GB/T 6565-2015)` in the following ways:

- For each top 1000 online job titles, we check what percentage of algorithm based mapping's most likely mapping has the same mapping results as manual mapping. Be more specific, we have mentioned `28,949,287 observations share the 1000 most frequent job posting titles'. 

  Let's say there are 1,000 postings have online job title "sale agent", and we manually mapped it to "salesman" in the GB/T 6565-2015. On the other hand, by algorithm 30% of "sale agent" can be mapped to "teacher", 20% can be mapped to "cook", 50% can be mapped to "salesman". Then, in this case "salesman" is the most likely mapping and it matches with the manual mapping result. We calculate the percentage of this chance for all the top 1000 online job titles. 

- We compare the frequency distribution of 28,949,287 observations for manual based mapping and algorithm based mapping. This is, we compare the GB/T 6565-2015 title frequency distribution based on manual and algorithms. Ideally, both distributions should share similar patterns. 





## Extract information Process
We extract information in the online job descriptions by searching for the key words.


