"""
In this project, we use the dataset from kaggle commonlit competition
URL: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries
"""
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# import dataset
train = pd.read_csv('summaries_train.csv')
test = pd.read_csv('summaries_test.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize data
train_encode = tokenizer.batch_encode_plus(train['text'].tolist(), padding=True, truncation=True)
test_encode = tokenizer.batch_encode_plus(test['text'].tolist(), padding=True, truncation=True)

# Tensor Dataset
train_dataset = TensorDataset(
  torch.tensor(train_encode['input_ids']),
  torch.tensor(train_encode['attention_mask']),
  torch.tensor(train['content']),
  torch.tensor(train['wording'])
)

test_dataset = TensorDataset(
  torch.tensor(test_encode['input_ids']),
  torch.tensor(test_encode['attention_mask'])
)

# split train and val
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)