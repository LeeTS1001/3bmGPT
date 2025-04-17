from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
import pandas as pd
from glob import glob
import os

# Load the model configuration
config = GPT2Config.from_pretrained('/checkpoint/config.json')
model = GPT2LMHeadModel(config)

# Load the state dict
state_dict = torch.load(
    '/checkpoint/pytorch_model.bin',
    map_location=torch.device('cpu')
)
print(state_dict.keys())

# Load the model
model.load_state_dict(state_dict, strict=False)

# load tokenize dataset
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/tokenizer_file")
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

#Generation example for a sample word
samp_input_txt = '1_4_N:C_N_C-A_CB:N_CA_C'
input_ids = tokenizer.encode(samp_input_txt, return_tensors="pt") 

output_ids = model.generate(
    input_ids=input_ids,       # Input IDs
    max_length=2,             # Maximum length of the generated sequence
    num_return_sequences=1,    # Number of sequences to generate
    do_sample=True,            # Enable sampling (for diverse results)
    top_k=50,                  # Use top-k sampling
    top_p=0.95,                # Use nucleus sampling
    temperature=0.7            # Control randomness
)

# Decode the generated text
tokenizer.decode(output_ids[0], skip_special_tokens=True)
