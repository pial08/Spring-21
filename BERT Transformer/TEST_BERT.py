#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


# In[2]:


df = pd.read_csv("data/IMDB Dataset.csv")


# In[3]:


df


# In[4]:



def to_sentiment(sentiment):
    #print(sentiment)
    if sentiment == "positive":
        return 1
    else:
        return 0

df['sentiment'] = df.sentiment.apply(to_sentiment)


# In[5]:


print(len(df.review[0]))

df


# In[6]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[7]:


tokenizer


# In[8]:


tokens = tokenizer.tokenize(df.review[0])
token_ids = tokenizer.convert_tokens_to_ids(tokens)

encoding = tokenizer.encode_plus(
    df.review[0],
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding=True,
    return_attention_mask=True,
    return_tensors="pt"
    )
#print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))\
#encoding['attention_mask']


# In[9]:


"""
token_len = []
#encoding = tokenizer.encode_plus(df.review[0],max_length=512)

#print(encoding)
for txt in tqdm(df.review):
    tokens = tokenizer.encode(txt, max_length=512)
    token_len.append(len(tokens))
    
#token_len
"""


# In[10]:


#sns.distplot(token_len)
#plt.xlim([0, 512]);
#plt.xlabel('Token count');


# In[11]:


MAX_LEN = 256


# In[12]:


class IMDBReview(Dataset):
    
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]
        
        encoding = tokenizer.encode_plus(
            review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
        )
        return {
            'review_text' : review,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(), 
            'target' : torch.tensor(target, dtype=torch.long)
        }
   

      


# In[13]:


class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    print(type(target))
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


# In[14]:


RANDOM_SEED = 42
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


# In[15]:


df_test.shape, df_train.shape, df_val.shape


# In[16]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = IMDBReview(
        reviews = df.review.to_numpy(),
        targets = df.sentiment.to_numpy(), 
        tokenizer = tokenizer,
        max_len = max_len
    )
    
    return DataLoader(
      ds, 
      batch_size=batch_size, 
      num_workers = 4
     )


# In[17]:


BATCH_SIZE = 16


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[18]:


print(train_data_loader)
data = next(iter(train_data_loader))
data.keys()


# In[19]:


print(data.keys())


# In[20]:


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)


# In[21]:


last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'], 
    attention_mask=encoding['attention_mask'])


# In[22]:


last_hidden_state


# In[23]:


bert_model.config.hidden_size


# In[24]:


pooled_output.shape


# In[25]:


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# In[26]:


#temp
device = torch.device("cuda:0")
class_names = ["positive", "negative"]
model = SentimentClassifier(len(class_names))
model = model.to(device)


#from apex.parallel import DistributedDataParallel as DDP
#model = DDP(model)

n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    print("Number of GPU is ", n_gpu)
    model = torch.nn.DataParallel(model)
print(model)


# In[27]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape)
print(attention_mask.shape)


# In[28]:


torch.cuda.empty_cache()
F.softmax(model(input_ids, attention_mask), dim=1)


# In[29]:


EPOCHS = 3
#what is correct bias
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
#scheduler has linear relationship with the total number of steps and drops lr to 0 at end
scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,# lr goes from 0 to the initial value
                num_training_steps=total_steps
            )

loss_function = nn.CrossEntropyLoss().to(device)


# In[30]:


def train(
    model, 
    data_loader, 
    loss_function, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
):
    model = model.train()
    
    losses = []
    correct_pred = 0
    
    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['target'].to(device)
        
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask
                       )
        _, preds= torch.max(outputs, dim=1)
        loss = loss_function(outputs, targets)
        
        correct_pred += torch.sum(preds == targets)
        #what does loss item return???
        losses.append(loss.item())
        
        loss.backward()
        #understand this line
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_pred.double() / n_examples, np.mean(losses)
        


# In[31]:


def eval(model, 
        data_loader, 
        loss_function, 
        device, n_examples
        ):
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for data in data_loader:
            
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['target'].to(device)

            outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask
                       )
            _, preds= torch.max(outputs, dim=1)
            loss = loss_function(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            #what does loss item return???
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
            


# In[ ]:





# In[ ]:


#get_ipython().run_line_magic('time', '')

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print("Epoch - ", epoch + 1)
    torch.cuda.empty_cache()

    train_accuracy, train_loss = train(
        model, train_data_loader, 
        loss_function, 
        optimizer, 
        device, 
        scheduler, 
        len(df_train)
    )
    torch.cuda.empty_cache()
    print("train done-- starting validation")
    validation_accuracy, validation_loss = eval(
        model,
        val_data_loader, 
        loss_function, 
        device, 
        len(df_val)
    )
    
    print(f'Val   loss {validation_loss} accuracy {validation_accuracy}')
    
    history['train_acc'].append(train_accuracy)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(validation_accuracy)
    history['val_loss'].append(validation_loss)
    torch.cuda.empty_cache()


    PATH = 'best_model_state.bin'

    if validation_accuracy > best_accuracy:

        #torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = validation_accuracy
        torch.save({'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss_function,}, PATH)


# In[ ]:



test_accuracy, _ = eval(model,
                        test_data_loader, 
                        loss_function, 
                        device,
                        len(df_test)
                       )

print(test_accuracy.item())


