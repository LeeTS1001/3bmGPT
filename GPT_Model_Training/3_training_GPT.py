from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import copy
import re
import datasets
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import json
import os
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# Load token and tokenized data
tokenized_dataset = load_from_disk('/tokenized_data')

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/space_bpe")
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired

# Set PAD token
if not tokenizer.pad_token:
    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token = "[PAD]"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Set config
config = GPT2Config(
    vocab_size=len(tokenizer),  # Ensure 'custom_tokenizer' is defined earlier
    n_positions=1024,
    n_ctx=1024,
    n_embd=1024,  # Reduced from 1024 for GPT-2 medium
    n_layer=24,  # Reduced from 24 for GPT-2 medium
    n_head=16   # Reduced from 16 for GPT-2 medium
)

# Instantiate a new GPT-2 model
model = GPT2LMHeadModel(config)

model.tokenizer = tokenizer
model.resize_token_embeddings(len(tokenizer))

# Set training parameters
training_args = TrainingArguments(
    output_dir="/mnt/prj/Taesub/Filtered_data_gpt2_0618", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output directory
    num_train_epochs=3, # Number of training epochs
    per_device_train_batch_size=32, # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    max_steps=68728, # Total training steps
    eval_steps=500, # Evaluate every 500 steps
    save_steps=3500, # Save model every 3500 steps
    logging_steps=2000, # Log every 2000 steps
    evaluation_strategy='steps', # Use steps for evaluation strategy
    learning_rate=1e-4, # Learning rate
    weight_decay=0.01, # Weight decay
    warmup_steps=500, # Number of warmup steps for learning rate
)

# Set data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator
)

trainer.train()

# Save the model
model.save_pretrained("/gpt2_model")

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
