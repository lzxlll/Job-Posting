## The `LSTM` model
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer




tokenizer = BertTokenizer.from_pretrained("PATH/chinese-bert-wwm/")
batch_size = 250

df = pd.read_csv("PATH/est_sample.csv", encoding = "utf_8_sig", on_bad_lines='skip', encoding_errors='ignore')

# replace the symbol '-' to '.' in soc_code column, and convert soc_code to int
df['soc_code'] = df['soc_code'].str.replace('-', '')
# replace 'Yes' with True and NaN with False using the fillna() and astype() methods
df['true_ind'] = df['true_ind'].fillna(False).astype(bool)

# generate a new column 'soc_code1' with value to recode the 'soc_code' in ascending order
# Create a dictionary to map unique soc_codes to sequential integer labels
unique_soc_codes = sorted(df['soc_code'].unique())
soc_code_dict  = {soc_code: i for i, soc_code in enumerate(unique_soc_codes)}
# Load datasets from CSV files
train_df_sample = pd.read_csv('PATH/train_df_sample.csv', encoding="utf_8_sig")
valid_df_sample = pd.read_csv('PATH/valid_df_sample.csv', encoding="utf_8_sig")
test_df_sample = pd.read_csv('PATH/test_df_sample.csv', encoding="utf_8_sig")


# drop index column, 'true_ind' and 'sample' columns
train_df_sample = train_df_sample.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code'
train_df_sample['soc_code'] = train_df_sample['soc_code'].astype(str)
train_df_sample['soc_code1'] = train_df_sample['soc_code'].map(soc_code_dict)

# drop index column, 'true_ind' and 'sample' columns
test_df_sample = test_df_sample.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code' for the test set
test_df_sample['soc_code'] = test_df_sample['soc_code'].astype(str)
test_df_sample['soc_code1'] = test_df_sample['soc_code'].map(soc_code_dict)

# drop index column, 'true_ind' and 'sample' columns
valid_df_sample = valid_df_sample.drop(['Unnamed: 0'], axis = 1)
# Generate a new column 'soc_code1' with the mapped values from 'soc_code' for the validation set
valid_df_sample['soc_code'] = valid_df_sample['soc_code'].astype(str)
valid_df_sample['soc_code1'] = valid_df_sample['soc_code'].map(soc_code_dict)




# Tokenize the text and convert it into input features
train_titles = train_df_sample['工作名称'].astype(str).tolist()
train_texts = train_df_sample['工作描述'].astype(str).tolist()
train_labels = train_df_sample['soc_code1'].tolist()

test_titles = test_df_sample['工作名称'].astype(str).tolist()
test_texts = test_df_sample['工作描述'].astype(str).tolist()
test_labels = test_df_sample['soc_code1'].tolist()

valid_titles = valid_df_sample['工作名称'].astype(str).tolist()
valid_texts = valid_df_sample['工作描述'].astype(str).tolist()
valid_labels = valid_df_sample['soc_code1'].tolist()










def assign_weight(true_ind):
    if true_ind:
        return 1.0
    else:
        return 0.5
    
# Assuming you have a column called 'true_ind' in your dataset with boolean values
# True for credible labels and False for less credible labels
train_df_sample['weight'] = train_df_sample['true_ind'].apply(assign_weight)
test_df_sample['weight'] = test_df_sample['true_ind'].apply(assign_weight)
valid_df_sample['weight'] = valid_df_sample['true_ind'].apply(assign_weight)

# Extract the weights
train_weights = train_df_sample['weight'].tolist()
valid_weights = valid_df_sample['weight'].tolist()
test_weights = test_df_sample['weight'].tolist()







from torch.utils.data import Dataset

# Create the JobPostingDataset class
class JobPostingDataset(Dataset):
    def __init__(self, titles, descriptions, labels, weights, tokenizer, max_length):
        self.titles = titles
        self.descriptions = descriptions
        self.labels = labels
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]
        weight = self.weights[idx]

        # Concatenate title and description, repeat the title to give it more importance
        repeat_title = 2  # Adjust this value to control the importance of the title
        text = (title + " ") * repeat_title + description

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Return a tuple of the input tensors, label, and weight
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )




import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Create the datasets
max_length = 512
train_dataset = JobPostingDataset(train_titles, train_texts, train_labels, train_weights, tokenizer, max_length)
valid_dataset = JobPostingDataset(valid_titles, valid_texts, valid_labels, valid_weights, tokenizer, max_length)
test_dataset = JobPostingDataset(test_titles, test_texts, test_labels, test_weights, tokenizer, max_length)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)






### Step 2: Model Definition and Steps for Grid Search
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import itertools


# Assuming that train_labels is a list or numpy array of your training labels
# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)


# Assuming train_loader, valid_loader, and test_loader are already defined
epochs = 10
best_valid_loss = float('inf')
early_stopping_patience = 3
early_stopping_counter = 0

# LSTM Model Definition
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout=0.2):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Hyperparameters (Assuming word_index and other data related variables are already defined)
vocab_size = len(tokenizer)  # Replace with your actual vocab size
embed_dim = 100
num_layers = 2
num_classes = 406  # Replace with your actual number of classes
bidirectional = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# Define hyperparameter space
hidden_dims = [25, 50, 100]
dropouts = [0.1, 0.2, 0.5]

# Grid search
best_accuracy = 0
best_hyperparams = None
best_model_state = None

for hidden_dim, dropout in itertools.product(hidden_dims, dropouts):
    # Initialize the model
    model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_layers, num_classes, bidirectional, dropout)

    # Check if multiple GPUs are available and wrap the model using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Move the model to the primary device
    model.to(device)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch in train_loader:
            inputs, _, labels, sample_weights = batch
            inputs, labels = inputs.to(device), labels.to(device)
            sample_weights = sample_weights.to(device)  # Move sample weights to the same device as the model

            optimizer.zero_grad()
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss per sample
            weighted_loss = loss * sample_weights  # Apply weights
            final_loss = weighted_loss.mean()  # Take mean of weighted loss
            final_loss.backward()  # Backpropagation
            optimizer.step()
            train_loss += final_loss.item()

        # Validation phase
        valid_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                inputs, _, labels, _ = batch  # Unpack and ignore attention mask and weights
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        validation_accuracy = correct / total

    # Store the best hyperparameters and model state
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_hyperparams = (hidden_dim, dropout)
        best_model_state = model.state_dict()

# Save the best model
print(f"Best Hidden Dim: {best_hyperparams[0]}, Best Dropout: {best_hyperparams[1]}, Best Accuracy: {best_accuracy}")
model_save_path = 'PATH/lstm_model_state_dict.pth'
torch.save({
    'state_dict': best_model_state,
    'hyperparams': best_hyperparams
}, model_save_path)


### Load the trained model (from `HPC`) to perform evaluation
# Assuming you have the model class defined (e.g., TextLSTM)

# Load the trained model and hyperparameters
checkpoint = torch.load('PATH/lstm_model_state_dict.pth', map_location=torch.device('cpu'))
hidden_dim, dropout = checkpoint['hyperparams']
state_dict = checkpoint['state_dict']

model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_layers, num_classes, bidirectional, dropout)

# Adjust for the keys if it was saved with nn.DataParallel
if list(state_dict.keys())[0].startswith('module.'):
    # Remove 'module.' prefix
    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

# Load the adjusted state dict
model.load_state_dict(state_dict)


from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Create a reverse mapping dictionary to convert the sequential labels back to SOC labels
reverse_soc_code_dict = {v: k for k, v in soc_code_dict.items()}

# Evaluate the model and store predictions
model = model.to(device)
model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for inputs, _, labels, _ in test_loader:  # Ignore attention masks
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

# Convert predictions and true_labels into NumPy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Assuming 'reverse_soc_code_dict' is a dictionary that maps the numeric labels back to the original SOC codes
soc_predictions = [reverse_soc_code_dict[p] for p in predictions]
soc_true_labels = [reverse_soc_code_dict[l] for l in true_labels]

# Compute classification report
unique_labels = sorted(list(set(soc_true_labels).union(set(soc_predictions))))
report = classification_report(soc_true_labels, soc_predictions, labels=unique_labels, output_dict=True)

# Convert report to a pandas DataFrame
report_df = pd.DataFrame(report).transpose()

# Print the report
print(report_df)

# Save the report to a CSV file
report_df.to_csv('PATH/lstm_report_df.csv', index=True)
