# Job-Posting
The total number of observations we acquired from raw job posting data is 140,107,287.

This project intends to:
1. identify and drop the duplicated/missing value job vacancy postings (get_surplus_sample.py, delete_duplicate.py).
2. extract the 1000 most frequent job posting titles, then manually map to the standard Chinese occupation codes (extract_the_first_1000_job_title.py). 
3. mapping the posting job titles to standard Chinese occupation codes.
4. assess the quality of the mapping by comparing mapping of the 1000 most frequent job postings by manual to the mapping by algorithm. 
5. extract information from job descriptions. 


# Identify and drop the duplicated/missing value job vacancy postings
We first delete the observations with missing job title information (obs = 1,305,020), this accounts for 0.9314% of the total observations. 
Second, we drop the duplicated job postings following the creteria: we sort the observations with same firm ID and job title in an ascending date order, the duplicated job postings refers to postings with time gap less than 1 month. This process drops 18,497,606 which amounts to 13.2025% of the total observations.


# Extract the 1000 most frequent job posting titles
To identify the 1000 most frequent job posting titles, we group the postings by job titles. Then, calculate the frequency within each group. There are 27,674,229 observations share the 1000 most frequent job posting titles, which amounts to 19.7522% of the total observations.
