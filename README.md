# Job-Posting
The raw job posting data has a total number of observations of 140,107,287.

This project intends to:
1. Clean the raw data.
2. Label the sample data used for fine-tune.
3. Fine-tune the BERT-WWM model.
4. Evaluate the performance. 

## ***Summary of Data Cleaning***

#### 1. **Identify and drop the duplicated/missing value job vacancy postings**
We first delete the observations with missing job title information (obs = ?), this accounts for ?% of the total observations. We have the following informations:

| Variable | Format | Variable | Format | Variable | Format |
| --- | --- |  --- | --- |   --- | --- |
|招聘主键ID  | bigint(20)| 工作薪酬 | varchar(255)| 工作名称 | varchar(255)|
|公司ID  | bigint(20)| 教育要求 | varchar(255)| 招聘数量 | varchar(255)|
|公司名称 | varchar(255)| 工作经历 | varchar(255)|发布日期  | datetime|
|城市名称 | varchar(255)| 工作描述 | varchar(255)|行业名称 | varchar(255)|
|公司所在区域 | varchar(255)| 职位名称 | varchar(255)|来源 | varchar(255)|

### Data Generation and Cleaning Process for Job Posting Data

In this section, we outline the data generation and cleaning process for job posting data used in our analysis. The primary data consists of job postings collected from 10 major job posting websites in China. The data is cleaned and processed to remove duplicates, unnecessary entries, and inconsistencies to ensure reliability and accuracy in our analysis.

#### 1. Data Import and Preliminary Cleaning
The raw data files are imported as pandas DataFrames using the `read_dat_data` function. This function reads the `.dat` files and assigns appropriate column names to the DataFrames.

#### 2. Data Preprocessing
- Missing values in the `工作名称` (job title) column are replaced with values from the `职位名称` (position name) column.
- Rows with missing `发布日期` (publish date) are dropped.
- Entries with `兼职` (part-time) in the job title are dropped.
- The dataset is filtered to include only data from 10 major job posting websites in China.
- Duplicate entries within a month, with the same `公司ID` (company ID), `工作名称` (job title), and `城市名称` (city name) are removed.

#### 3. Data Separation
After preprocessing, the `工作描述` (job description) column is separated from the rest of the data to make the dataset more manageable.

#### 4. Data Aggregation
The cleaned and preprocessed data is saved as separate CSV files. All the CSV files are combined into a single DataFrame, containing only the `工作名称` (job title) column. This final DataFrame is saved as `charac_posting.csv`, which is used for further analysis.




## ***Summary of Data Processing***

In this study, we aim to analyze job postings data to classify them into Standard Occupational Classification (SOC) categories. The data generation and cleaning process is as follows:

1. **Read the data**:  
   Load the raw job posting data from a CSV file into a pandas DataFrame, **`df`**. Only the '工作名称' (job title) column is needed for this step.

2. **Filter out rare job titles**:  
   Count the occurrences of each unique job title in the DataFrame, and filter out those that occur less than 5 times, resulting in 883,695 unique job titles across 50,340,840 job postings.

3. **Parallelize job title classification**:  
   Split the DataFrame containing the filtered job titles into 300 smaller DataFrames and save each sub-DataFrame as a CSV file.

4. **Classify job titles using ChatGPT**:  
   Define a function, **`classify_job_title`**, that takes a job title and an API key, and returns the most likely SOC code for that job title. The function sends a request to the ChatGPT API using the provided API key and extracts the SOC code from the API response.

5. **Run classification in parallel**:  
   Using Python's `ThreadPoolExecutor`, create a pool of 30 worker threads to classify the job titles concurrently. Read each sub-DataFrame containing job titles from the CSV files, submit the job titles to the thread pool for classification, append the resulting SOC codes to the DataFrame, and save it as a new CSV file.

6. **Merge the classified titles**:  
   Append all the classified titles' CSV files to create a single DataFrame, **`df_title`**. Keep only the '工作名称' and 'soc_code' columns, and drop any rows with missing SOC codes.

7. **Map unmapped job postings using job descriptions**:  
   Merge the DataFrame containing the SOC codes with another DataFrame containing job descriptions based on the job title. This will be used to map the remaining unmapped job postings.

8. **Load ONET SOC job titles**:  
   Load a DataFrame containing all possible SOC job titles from the ONET dataset to help remove incorrect mappings.

9. **Filter out rare SOC codes**:  
   Filter out broad occupations with less than 100 job postings to ensure that there is enough data to train a good model. This process results in 408 broad occupations.

10. **Randomly sample job postings**:  
    Randomly sample 3,000 job postings within each broad occupation and save them to a CSV file. Feed this data to ChatGPT to map the job postings to SOC categories using job descriptions.

11. **Verify labeling based on job descriptions**:  
    Define a function, **`classify_job_desp`**, that takes a job description, job title, and API key, and returns whether the given SOC code is a reasonable classification based on the job description. The function sends a request to the ChatGPT API using the provided API key and returns a yes or no answer.

12. **Double-check the sub-sampled dataset**:  
    Parallelize the job description classification using `ThreadPoolExecutor`, and append the true/false indicator to the DataFrame. Save the resulting DataFrame as a new CSV file.

13. **Generate the final dataset for model fine-tuning**:  
    Dataset that passed the second check (classification using job descriptions). This final dataset is used for model fine-tuning.





## ***Summary of Fine-tune Chinese BERT-wwm***

In this Python code, we are fine-tuning a pre-trained BERT model to classify job postings based on their Standard Occupational Classification (SOC) codes using the `transformers` library. Below is a breakdown of the code into several steps with detailed explanations for each.

1. **Import necessary libraries and load data**:  
   First, we import essential libraries such as `pandas`, `torch`, and `transformers`, and read the data from a CSV file. The data includes columns like `soc_code`, `true_ind`, `工作名称` (job title), and `工作描述` (job description).

2. **Preprocessing the data**:  
   We clean the `soc_code` column by removing '-' symbols and converting it to integers. In the `true_ind` column, we replace 'Yes' with `True` and `NaN` with `False`, indicating if a label is more reliable. This reliability indicator will be used to assign different weights to samples during model training.

3. **Splitting the data**:  
   We split the data into training, validation, and testing sets in a 60-20-20 ratio using the `train_test_split` function from the `scikit-learn` library.

4. **Creating a new column 'soc_code1'**:  
   We generate a new column, `soc_code1`, which maps unique SOC codes to sequential integer labels. This simplifies the classification task and aids the model in learning patterns in the data.

5. **Tokenization**:  
   Job titles and descriptions are tokenized using the BERT tokenizer, which converts text into a format compatible with the BERT model.

6. **Creating a custom PyTorch Dataset**:  
   We create a custom PyTorch Dataset class, `JobPostingDataset`, to store the tokenized text, labels, and weights. This class will be used to generate data loaders for efficient training, validation, and testing.

7. **Preparing data loaders**:  
   Using the custom `JobPostingDataset` class, we create data loaders that enable efficient batch loading of data during training, validation, and testing.

8. **Computing class weights**:  
   To address class imbalance, we calculate class weights from the training data and incorporate them into the `CrossEntropyLoss` criterion. This encourages the model to give more attention to minority classes during training.

9. **Defining the model, optimizer, and learning rate scheduler**:  
    We fine-tune a pre-trained BERT model for our classification task, setting the output label count to the number of unique `soc_code1` values. The `AdamW` optimizer and a learning rate scheduler with a warmup period are used for training.

10. **Training the model with early stopping**:  
    The model is trained using a loop with early stopping based on validation loss. If validation loss doesn’t improve for a specified number of consecutive epochs (as set by the `early_stopping_patience` variable), training halts to prevent overfitting.

11. **Evaluating the model**:  
    The model's performance is evaluated on the validation set using a custom evaluation function that computes the validation loss. The model with the lowest validation loss is saved as the best model.


