import pandas as pd
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
from torch.utils.data import DataLoader

# Create the datasets
max_length = 512
train_dataset = JobPostingDataset(train_titles, train_texts, train_labels, train_weights, tokenizer, max_length)
valid_dataset = JobPostingDataset(valid_titles, valid_texts, valid_labels, valid_weights, tokenizer, max_length)
test_dataset = JobPostingDataset(test_titles, test_texts, test_labels, test_weights, tokenizer, max_length)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)







# Calculate Class Weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming 'train_labels' is a list of your training labels
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Modify Loss Function
# Define the weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import itertools


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=32, kernel_size=3)

        # Global Max Pooling Layer
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Intermediate Dense Layer with 250 hidden units
        self.fc1 = nn.Linear(32, 250)

        # Output Layer with softmax activation for classification
        self.fc2 = nn.Linear(250, num_classes)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x).permute(0, 2, 1)  # Rearrange to [batch, channels, sequence length]

        # Convolution and ReLU activation
        x = F.relu(self.conv1d(x))

        # Global Max Pooling
        x = self.global_max_pool(x).squeeze(2)

        # Intermediate Dense Layer
        x = F.relu(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        # Softmax activation
        return F.log_softmax(x, dim=1)

# Hyperparameters for grid search
hidden_dims = [25, 50, 100]
dropouts = [0.1, 0.2, 0.5]  # Assuming you're adding dropout to your model

# Other fixed hyperparameters
vocab_size = len(tokenizer)  # Replace with your actual vocab size
embed_dim = 100
num_classes = 406  # Replace with your actual number of classes
epochs = 10

# Grid search
best_hyperparams = {'hidden_dim': None, 'dropout': None}
best_valid_loss = float('inf')

for hidden_dim, dropout in itertools.product(hidden_dims, dropouts):
    print(f"Training with hidden_dim: {hidden_dim}, dropout: {dropout}")

    # Initialize the model with current hyperparameters
    model = TextCNN(vocab_size, embed_dim, num_classes)

    # Check if multiple GPUs are available and wrap the model using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Use the weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training and Validation loop
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        for inputs, _, labels, weights in train_loader:  # Ensure weights are returned by the loader
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)

            # Calculate loss without reduction
            loss = F.cross_entropy(outputs, labels, reduction='none')
            # Apply sample weights
            weighted_loss = (loss * weights).mean()  # Average the weighted loss
            
            # Backward pass and optimize
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, _, labels, _ in valid_loader:  # Ignore attention mask and weights
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    # Update best hyperparameters and save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_hyperparams['hidden_dim'] = hidden_dim
        best_hyperparams['dropout'] = dropout

        # Save the best model
        torch.save(model.state_dict(), 'PATH/cnn_model_state_dict.pth')  # Update path as necessary

print(f"Best Hyperparameters: Hidden Dim - {best_hyperparams['hidden_dim']}, Dropout - {best_hyperparams['dropout']}")


from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


model = TextCNN(vocab_size, embed_dim, num_classes)
# Load the saved state dict for the CNN model
cnn_state_dict = torch.load('PATH/cnn_model_state_dict.pth')

# Adjust for the keys if it was saved with nn.DataParallel
if list(cnn_state_dict.keys())[0].startswith('module.'):
    cnn_state_dict = {key[len("module."):]: value for key, value in cnn_state_dict.items()}

# Load the adjusted state dict into your CNN model
model.load_state_dict(cnn_state_dict)

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

# Debug: Print the lengths of predictions and true_labels
print(f"Length of predictions: {len(predictions)}")
print(f"Length of true_labels: {len(true_labels)}")

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
report_df.to_csv('PATH/cnn_report_df.csv', index=True)