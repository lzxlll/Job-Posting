import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
bert_model = BertModel.from_pretrained("/share/home/320346/chinese-bert-wwm/")
tokenizer = BertTokenizer.from_pretrained("/share/home/320346/chinese-bert-wwm/")

df = pd.read_csv("/share/home/320346/est_sample.csv", encoding = "utf_8_sig", on_bad_lines='skip', encoding_errors='ignore')

# replace the symbol '-' to '.' in soc_code column, and convert soc_code to int
df['soc_code'] = df['soc_code'].str.replace('-', '')
# replace 'Yes' with True and NaN with False using the fillna() and astype() methods
df['true_ind'] = df['true_ind'].fillna(False).astype(bool)

# generate a new column 'soc_code1' with value to recode the 'soc_code' in ascending order
# Create a dictionary to map unique soc_codes to sequential integer labels
unique_soc_codes = sorted(df['soc_code'].unique())
soc_code_dict  = {soc_code: i for i, soc_code in enumerate(unique_soc_codes)}
# Load datasets from CSV files
train_df = pd.read_csv('/share/home/320346/train_df_sample.csv', encoding="utf_8_sig")
valid_df = pd.read_csv('/share/home/320346/valid_df_sample.csv', encoding="utf_8_sig")
test_df = pd.read_csv('/share/home/320346/test_df_sample.csv', encoding="utf_8_sig")

# drop index column, 'true_ind' and 'sample' columns
train_df = train_df.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code'
train_df['soc_code'] = train_df['soc_code'].astype(str)
train_df['soc_code1'] = train_df['soc_code'].map(soc_code_dict)

# drop index column, 'true_ind' and 'sample' columns
test_df = test_df.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code' for the test set
test_df['soc_code'] = test_df['soc_code'].astype(str)
test_df['soc_code1'] = test_df['soc_code'].map(soc_code_dict)

# drop index column, 'true_ind' and 'sample' columns
valid_df = valid_df.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code' for the validation set
valid_df['soc_code'] = valid_df['soc_code'].astype(str)
valid_df['soc_code1'] = valid_df['soc_code'].map(soc_code_dict)

# Combine titles and descriptions for tokenization
combined_texts_train = (train_df['工作名称'] + train_df['工作名称'] + " " + train_df['工作描述']).astype(str).tolist()
combined_texts_valid = (valid_df['工作名称'] + valid_df['工作名称'] + " " + valid_df['工作描述']).astype(str).tolist()
combined_texts_test = (test_df['工作名称'] + test_df['工作名称'] + " " + test_df['工作描述']).astype(str).tolist()

def assign_weight(true_ind):
    if true_ind:
        return 1.0
    else:
        return 0.5
    
# Assuming you have a column called 'true_ind' in your dataset with boolean values
# True for credible labels and False for less credible labels
train_df['weight'] = train_df['true_ind'].apply(assign_weight)
test_df['weight'] = test_df['true_ind'].apply(assign_weight)
valid_df['weight'] = valid_df['true_ind'].apply(assign_weight)

# Extract the weights
train_weights = train_df['weight'].tolist()
valid_weights = valid_df['weight'].tolist()
test_weights = test_df['weight'].tolist()
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

def get_bert_embeddings(sentences, tokenizer, model):
    embeddings = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the embeddings from the last hidden state
        last_hidden_states = outputs.last_hidden_state
        # Average the token embeddings for the sentence
        sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
        embeddings.append(sentence_embedding)
    
    return embeddings

# Example for training data
train_embeddings = get_bert_embeddings(combined_texts_train, tokenizer, bert_model)
valid_embeddings = get_bert_embeddings(combined_texts_valid, tokenizer, bert_model)
test_embeddings = get_bert_embeddings(combined_texts_test, tokenizer, bert_model)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Convert embeddings to numpy arrays for sklearn compatibility
train_features = np.vstack(train_embeddings)
valid_features = np.vstack(valid_embeddings)
test_features = np.vstack(test_embeddings)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [20, 50, 100, 200]
}

# Initialize Random Forest model

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['soc_code1']), y=train_df['soc_code1'])
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

rf_model = RandomForestClassifier(random_state=42, n_jobs=18, class_weight=class_weight_dict)

# Set up Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=18)

# Train the model
grid_search.fit(train_features, train_df['soc_code1'], sample_weight=train_weights)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
import pickle

# Assuming 'best_rf_model' is your fine-tuned model from GridSearchCV
model_file_path = '/share/home/320346/random_forest_model.pkl'

# Save the model to the file
with open(model_file_path, 'wb') as file:
    pickle.dump(best_rf_model, file)
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Convert test embeddings to numpy array
test_features = np.vstack(test_embeddings)

# Make predictions on the test data
predictions = best_rf_model.predict(test_features)

# Assuming 'test_df' is your test dataset and it contains the true labels in 'soc_code1'
true_labels = test_df['soc_code1'].values

# Create a reverse mapping dictionary to convert the numeric labels back to SOC labels
reverse_soc_code_dict = {v: k for k, v in soc_code_dict.items()}

# Convert numeric predictions and true_labels into original SOC codes
soc_predictions = [reverse_soc_code_dict.get(p, "Unknown") for p in predictions]
soc_true_labels = [reverse_soc_code_dict.get(l, "Unknown") for l in true_labels]

# Compute classification report
unique_labels = sorted(list(set(soc_true_labels + soc_predictions)))
report = classification_report(soc_true_labels, soc_predictions, labels=unique_labels, output_dict=True, zero_division=0)

# Convert report to a pandas DataFrame
report_df = pd.DataFrame(report).transpose()

# Print the report
print(report_df)

# Save the report to a CSV file
report_df.to_csv('/share/home/320346/rf_report_df.csv', index=True)