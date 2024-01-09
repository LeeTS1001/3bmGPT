from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import json
import copy
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


tokenized_dataset = load_from_disk('/mnt/prj/AJ/dock_bert/0521_GPT2/tokenized_dataset')

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/mnt/prj/AJ/dock_bert/0521_GPT2/BPE_data_all_tokenizer/save_bpe.json")
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired

if not tokenizer.pad_token:
    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token = "[PAD]"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

# Assuming your tokenized_dataset is a dictionary with "train", "eval" and "test" datasets
# Each of these datasets have the keys "input_ids" and "attention_mask"

# Create a configuration for GPT-2 medium
config = GPT2Config(
    vocab_size=len(tokenizer), # depends on your tokenizer
    n_positions=1024,
    n_ctx=1024,
    n_embd=1024,  
    n_layer=24,   
    n_head=16,    
)

# Instantiate a new GPT-2 model
model = GPT2LMHeadModel(config)

model.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))

# Set training arguments (You can keep these the same or adjust as needed for the medium model)
training_args = TrainingArguments(
    output_dir="/mnt/prj/AJ/dock_bert/0521_GPT2/gpt2_medium", # Adjusted the output directory to gpt2_medium
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,
    max_steps=2810238,
    eval_steps=2000,
    save_steps=100000, 
    logging_steps=2000,
    evaluation_strategy='steps',
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/mnt/prj/AJ/dock_bert/0521_GPT2/gpt2_medium")

# After the training
log_history = trainer.state.log_history

# Extract the losses from the log history
train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
eval_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]

# Save to a file
with open('train_history.json', 'w') as f:
    json.dump({
        'train_loss': train_losses,
        'eval_loss': eval_losses
    }, f)
