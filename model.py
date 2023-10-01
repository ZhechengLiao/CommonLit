# Build Model
from transformers import BertModel
import torch.nn as nn

class BERT_Model(nn.Module):
  def __init__(self):
    super(BERT_Model, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased') # use bert pretrained on hugging face
    self.drop = nn.Dropout(p=0.1) # dropout layer to avoid overfit
    
    # after bert, we make a fully connect layer to do the final regression output
    self.ffc = nn.Sequential(
      nn.Linear(768, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 2),
    )

  def __call__(self, input_ids, attention_mask):
    output = self.bert(input_ids, attention_mask)
    output = output.pooler_output
    output = self.drop(output)
    output = self.ffc(output)
    return output.double()