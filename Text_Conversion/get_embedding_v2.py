#!/usr/bin/env python
# coding: utf-8

# In[3]:


from glob import glob
import os
import sys
import pandas as pd
from tqdm import tqdm
import json
import copy
import random
import re
import datasets
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling

### version 231102

### operation : conda activate Dock_gpt2 ; cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python get_embedding_v2.py [model_parameter_path] [model_data_path] [tokenizer_file_path] [sentence_data_path]
# example : conda activate Dock_gpt2 ; cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python get_embedding_v2.py /ssd002/hhjung/3bmGPT_hhjung_230921/model/params_train_to_hf.json /ssd002/hhjung/3bmGPT_hhjung_230921/model/checkpoint_29964_to_hf.bin /ssd002/hhjung/3bmGPT_hhjung_230921/SB_BPE_tokenizer_char.json /ssd002/hhjung/3bmGPT_hhjung_230921/output_sentence/5edq_A_rec_5edq_5n3_lig_it1_it2_tt_1_complex_sentence.txt


# ## Load tokenizer & pretrained Model

# In[2]:


# load tokenize dataset
GPT_dir = "/ssd002/hhjung/3bmGPT_hhjung_230921" ###
tokenized_dataset = load_from_disk(GPT_dir + "/tokenized_dataset") ###

tokenizer_path = sys.argv[3] #/ssd002/hhjung/3bmGPT_hhjung_230921/SB_BPE_tokenizer_char.json
#tokenizer = PreTrainedTokenizerFast(tokenizer_file= GPT_dir + "/SB_BPE_tokenizer_char.json") ###
tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_path) ###
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# load model
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

## Load model
model_dir = GPT_dir + "/model"
model_parameter_path = sys.argv[1] #/ssd002/hhjung/3bmGPT_hhjung_230921/model/params_train_to_hf.json
config = GPT2Config.from_pretrained(model_parameter_path) ###
#config = GPT2Config.from_pretrained(model_dir + "/params_train_to_hf.json") ###

model = GPT2LMHeadModel(config)
model_data_path = sys.argv[2] #/ssd002/hhjung/3bmGPT_hhjung_230921/model/checkpoint_29964_to_hf.bin
model_state_dict = torch.load(model_data_path)
#model_state_dict = torch.load(model_dir + "/checkpoint_29964_to_hf.bin")
model.load_state_dict(model_state_dict)

# Instantiate a new GPT-2 model
# model = GPT2LMHeadModel(config)

model.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))


# In[4]:


print(os.getcwd())


# In[5]:


# file_path = "/ssd002/hhjung/3bmGPT_hhjung_230921/word_list/merged_raw_wordset_tSNE_10k_cerebras1.txt"  # Path to the text file ###
sentence_data_path = sys.argv[4] ###
file_path = sentence_data_path ###
with open(file_path, 'r') as file:
    lines = file.readlines()  # Read all lines of the file

# Now you can work with the lines of the file
merged_row= []
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace and newline characters
    merged_row.append(line)


# In[6]:


len(merged_row)


# In[11]:


# 50257 vector
embeddings = []
for sentence in tqdm(merged_row):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=1024)
    outputs = model(**inputs)
    # Get the hidden states and mean pool across the sequence length (1st) dimension
    embeddings.append(outputs.logits.mean(dim=1).detach().numpy())


# In[ ]:





# In[7]:


# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[11]:



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the device
model = model.to(device)

# Assuming you have a fixed batch size, but you can adjust accordingly
#BATCH_SIZE = 32

embeddings = []

# Create batches of sentences
#batches = [merged_row[i:i + BATCH_SIZE] for i in range(0, len(merged_row), BATCH_SIZE)]

for sentence in tqdm(merged_row):
    # Tokenize the entire batch at once
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=1024)

    # Move the tokenized inputs to the same device as the model (e.g., GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # This will disable gradient computations, making inference faster
        outputs = model(**inputs)
        
    # Get the hidden states and mean pool across the sequence length (1st) dimension
    #batch_embeddings = outputs.logits.mean(dim=1)
    
    #embeddings.append(batch_embeddings.cpu().numpy())
    
    #embeddings.append(outputs.logits.mean(dim=1).detach().numpy())
    embeddings.append(outputs.logits.mean(dim=1).cpu().detach().numpy())

# Flatten the list of batches into a single list
#embeddings = [item for sublist in embeddings for item in sublist]


# In[12]:


import numpy as np

# Assuming embeddings is a list of your arrays
sentence_pattern = (".").join(sentence_data_path.split(".")[0:-1])
np_output_path = sentence_pattern + ".npz"
#/ssd002/hhjung/3bmGPT_hhjung_230921/word_list/merged_raw_wordset_tSNE_10k_cerebras1_1000.npz
#np.savez('merged_raw_wordset_tSNE_10k_cerebras1.npz', *embeddings)
np.savez(np_output_path, *embeddings)
print(np_output_path) ### check raw data

npz_path = np_output_path
data = np.load(npz_path)

with np.load(npz_path) as data:
    dfs = [pd.DataFrame(data[key]) for key in data.keys()]
merged_df = pd.concat(dfs, axis=0, ignore_index=True)

data.close()
print(merged_df)

output_embedding_path = sentence_pattern + "_embedding_50257.txt"
merged_df.to_csv(output_embedding_path, sep="\t", header=None)
print(output_embedding_path)
