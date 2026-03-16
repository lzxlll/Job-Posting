


### This is similar to hierarchical_classification. The difference is that this code allows for more than one GPU to speed up the training process. ###







from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModelForMaskedLM, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertForSequenceClassification
import torch
from transformers import get_scheduler
from transformers import Trainer
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import BertPreTrainedModel
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput


#os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"

tokenizer = BertTokenizer.from_pretrained("/share/home/320346/chinese-bert-wwm/")


train_df_sample = pd.read_csv("/share/home/320346/train_df_sample.csv", encoding = "utf_8_sig", on_bad_lines='skip', encoding_errors='ignore')
test_df_sample = pd.read_csv("/share/home/320346/test_df_sample.csv", encoding = "utf_8_sig", on_bad_lines='skip', encoding_errors='ignore')
train_df_sample = train_df_sample.sample(n= 1000, random_state= 1)
test_df_sample = test_df_sample.sample(n= 1000, random_state= 1)


# keep the overalp soc_code in train and test dataset
train_df_sample = train_df_sample[train_df_sample['soc_code'].isin(test_df_sample['soc_code'])]
test_df_sample = test_df_sample[test_df_sample['soc_code'].isin(train_df_sample['soc_code'])]

# replace the symbol '-' to '.' in soc_code column, and convert soc_code to int
train_df_sample['soc_code'] = train_df_sample['soc_code'].str.replace('-', '')
test_df_sample['soc_code'] = test_df_sample['soc_code'].str.replace('-', '')

# create a new column 'major_group' as the first two digits of soc_code
train_df_sample['major_group'] = train_df_sample['soc_code'].str[:2].astype(int)
test_df_sample['major_group'] = test_df_sample['soc_code'].str[:2].astype(int)

# create a new column 'minor_group' as the third and four digits of soc_code

train_df_sample['minor_group'] = train_df_sample['soc_code'].str[2:4].astype(int)
test_df_sample['minor_group'] = test_df_sample['soc_code'].str[2:4].astype(int)

# create a new column 'broad_group' as the first six digits of soc_code
train_df_sample['broad_group'] = train_df_sample['soc_code'].str[4:6].astype(int)
test_df_sample['broad_group'] = test_df_sample['soc_code'].str[4:6].astype(int)




# generate a new column 'soc_code1' with value to recode the 'soc_code' in ascending order
train_df_sample['major_group1'] = train_df_sample['major_group'].rank(method='dense').astype(int) - 1
test_df_sample['major_group1'] = test_df_sample['major_group'].rank(method='dense').astype(int) - 1

train_df_sample['minor_group1'] = train_df_sample['minor_group'].rank(method='dense').astype(int) - 1
test_df_sample['minor_group1'] = test_df_sample['minor_group'].rank(method='dense').astype(int) - 1

train_df_sample['broad_group1'] = train_df_sample['broad_group'].rank(method='dense').astype(int) - 1
test_df_sample['broad_group1'] = test_df_sample['broad_group'].rank(method='dense').astype(int) - 1



# convert '工作描述' to string
train_df_sample['工作描述'] = train_df_sample['工作描述'].astype(str)
test_df_sample['工作描述'] = test_df_sample['工作描述'].astype(str)




# Tokenize the text and convert it into input features
train_texts = train_df_sample['工作描述'].tolist()
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

test_texts = test_df_sample['工作描述'].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)



# Convert the input features into PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])

train_major_labels = torch.tensor(train_df_sample['major_group1'].tolist())
train_minor_labels = torch.tensor(train_df_sample['minor_group1'].tolist())
train_broad_labels = torch.tensor(train_df_sample['broad_group1'].tolist())

test_major_labels = torch.tensor(test_df_sample['major_group1'].tolist())
test_minor_labels = torch.tensor(test_df_sample['minor_group1'].tolist())
test_broad_labels = torch.tensor(test_df_sample['broad_group1'].tolist())



# Create a PyTorch DataLoader to iterate over the data during training
batch_size = 5

train_data = TensorDataset(train_inputs, train_masks, train_major_labels, train_minor_labels, train_broad_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(test_inputs, test_masks, test_major_labels, test_minor_labels, test_broad_labels)
test_loader = DataLoader(test_data, batch_size=batch_size)








from transformers import BertPreTrainedModel
from torch import nn
from transformers import BertModel

class HierarchicalBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.major_classifier = nn.Linear(config.hidden_size, num_major_labels)
        self.minor_classifier = nn.Linear(config.hidden_size, num_minor_labels)
        self.broad_classifier = nn.Linear(config.hidden_size, num_broad_labels)

    def forward(self, input_ids, attention_mask, major_labels=None, minor_labels=None, broad_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        major_logits = self.major_classifier(pooled_output)
        minor_logits = self.minor_classifier(pooled_output)
        broad_logits = self.broad_classifier(pooled_output)


        return major_logits, minor_logits, broad_logits
        
        
        
        






num_epochs = 4
num_training_steps = num_epochs * len(train_loader)


num_major_labels = len(train_df_sample['major_group1'].unique())
num_minor_labels = len(train_df_sample['minor_group1'].unique())
num_broad_labels = len(train_df_sample['broad_group1'].unique())

# Create separate BERT models for each level of the hierarchy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if there are at least 3 GPUs available
assert torch.cuda.device_count() >= 2, "At least 3 GPUs must be available"












# Move the data and labels to the same device as the models
train_inputs = train_inputs.to(device)
train_masks = train_masks.to(device)
train_major_labels = train_major_labels.to(device)
train_minor_labels = train_minor_labels.to(device)
train_broad_labels = train_broad_labels.to(device)

test_inputs = test_inputs.to(device)
test_masks = test_masks.to(device)
test_major_labels = test_major_labels.to(device)
test_minor_labels = test_minor_labels.to(device)
test_broad_labels = test_broad_labels.to(device)


major_model = BertForSequenceClassification.from_pretrained("/share/home/320346/chinese-bert-wwm/", num_labels=num_major_labels).to(device)
minor_model = BertForSequenceClassification.from_pretrained("/share/home/320346/chinese-bert-wwm/", num_labels=num_minor_labels).to(device)
broad_model = BertForSequenceClassification.from_pretrained("/share/home/320346/chinese-bert-wwm/", num_labels=num_broad_labels).to(device)



# Wrap the models with DataParallel
major_model = nn.DataParallel(major_model)
minor_model = nn.DataParallel(minor_model)
broad_model = nn.DataParallel(broad_model)

# Create separate optimizers and schedulers for each model
major_optimizer = AdamW(major_model.parameters(), lr=2e-5, eps=1e-8)
minor_optimizer = AdamW(minor_model.parameters(), lr=2e-5, eps=1e-8)
broad_optimizer = AdamW(broad_model.parameters(), lr=2e-5, eps=1e-8)

major_scheduler = get_linear_schedule_with_warmup(major_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
minor_scheduler = get_linear_schedule_with_warmup(minor_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
broad_scheduler = get_linear_schedule_with_warmup(broad_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)










criterion = nn.CrossEntropyLoss()
accumulation_steps = 4


# Train the major model
# Train the major model
for epoch in range(num_epochs):
    major_model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        major_labels = batch[2].to(device)

        major_optimizer.zero_grad()

        outputs = major_model(inputs, attention_mask=masks, labels=major_labels)
        loss = outputs.loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(major_model.parameters(), 1.0)

        if (num_batches + 1) % accumulation_steps == 0:
            major_optimizer.step()
            major_scheduler.step()
            major_optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    epoch_loss = total_loss / num_batches
    print(f"Major Model - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Train the minor model, using major model predictions
# Train the minor model
for epoch in range(num_epochs):
    minor_model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        major_labels = batch[2].to(device)
        minor_labels = batch[3].to(device)

        with torch.no_grad():
            major_outputs = major_model(inputs, attention_mask=masks)
            major_predictions = torch.argmax(major_outputs.logits, axis=1)

        filtered_indices = torch.where(major_predictions.to(device) == major_labels.to(device))[0]
        if len(filtered_indices) == 0:
            continue

        filtered_inputs = inputs[filtered_indices]
        filtered_masks = masks[filtered_indices]
        filtered_minor_labels = minor_labels[filtered_indices]

        minor_optimizer.zero_grad()

        outputs = minor_model(filtered_inputs, attention_mask=filtered_masks, labels=filtered_minor_labels)
        loss = outputs.loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(minor_model.parameters(), 1.0)

        if (num_batches + 1) % accumulation_steps == 0:
            major_optimizer.step()
            major_scheduler.step()
            major_optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    epoch_loss = total_loss / num_batches
    print(f"Minor Model - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Train the broad model, using major and minor model predictions
# Train the broad model
for epoch in range(num_epochs):
    broad_model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        major_labels = batch[2].to(device)
        minor_labels = batch[3].to(device)
        broad_labels = batch[4].to(device)

        with torch.no_grad():
            major_outputs = major_model(inputs, attention_mask=masks)
            major_predictions = torch.argmax(major_outputs.logits, axis=1)
            minor_outputs = minor_model(inputs, attention_mask=masks)
            minor_predictions = torch.argmax(minor_outputs.logits, axis=1)

        filtered_indices = torch.where((major_predictions.to(device) == major_labels) & (minor_predictions.to(device) == minor_labels))[0]
        if len(filtered_indices) == 0:
            continue

        filtered_inputs = inputs[filtered_indices]
        filtered_masks = masks[filtered_indices]
        filtered_broad_labels = broad_labels[filtered_indices]

        broad_optimizer.zero_grad()

        outputs = broad_model(filtered_inputs, attention_mask=filtered_masks, labels=filtered_broad_labels)
        loss = outputs.loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(broad_model.parameters(), 1.0)

        if (num_batches + 1) % accumulation_steps == 0:
            major_optimizer.step()
            major_scheduler.step()
            major_optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    epoch_loss = total_loss / num_batches
    print(f"Broad Model - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")





# Save the major model
major_model.module.save_pretrained("/share/home/320346/trained_model/major_model")

# Save the minor model
minor_model.module.save_pretrained("/share/home/320346/trained_model/minor_model")

# Save the broad model
broad_model.module.save_pretrained("/share/home/320346/trained_model/broad_model")




    
    
# Evaluate the performance of the model on the test set
major_model.eval()
minor_model.eval()
broad_model.eval()

major_predictions = []
minor_predictions = []
broad_predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)

        major_logits = major_model(inputs, attention_mask=masks).logits
        major_batch_predictions = torch.argmax(major_logits, axis=1).cpu().numpy()
        major_predictions.extend(major_batch_predictions)

        minor_logits = minor_model(inputs, attention_mask=masks).logits
        minor_batch_predictions = torch.argmax(minor_logits, axis=1).cpu().numpy()
        minor_predictions.extend(minor_batch_predictions)

        broad_logits = broad_model(inputs, attention_mask=masks).logits
        broad_batch_predictions = torch.argmax(broad_logits, axis=1).cpu().numpy()
        broad_predictions.extend(broad_batch_predictions)

from sklearn.metrics import accuracy_score

major_accuracy = accuracy_score(test_major_labels.cpu(), major_predictions)
minor_accuracy = accuracy_score(test_minor_labels.cpu(), minor_predictions)
broad_accuracy = accuracy_score(test_broad_labels.cpu(), broad_predictions)




print(f"Major Group Accuracy: {major_accuracy}")
print(f"Minor Group Accuracy: {minor_accuracy}")
print(f"Broad Group Accuracy: {broad_accuracy}")
    
    
    



