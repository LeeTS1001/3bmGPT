import json
import os
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
import numpy as np

# Open the JSON file for reading
with open('/filtered_data.json', 'r') as f:
    # Load the JSON data from the file
    data = json.load(f)

import random

# Shuffle the data
random.shuffle(data)

# Split the data into train, validation, and test sets
num_examples = len(data)
num_train = int(num_examples * 0.7)  # 70% for training
num_val = int(num_examples * 0.2)  # 20% for validation
num_test = num_examples - num_train - num_val  # rest for test

train_data = data[:num_train]
val_data = data[num_train:num_train+num_val]
test_data = data[num_train+num_val:]

# Save train, validation, and test sets to text files
with open("/text_data/train.txt", "w") as f:
    for item in train_data:
        f.write("%s\n" % item)

with open("/text_data/val.txt", "w") as f:
    for item in val_data:
        f.write("%s\n" % item)

with open("/text_data/test.txt", "w") as f:
    for item in test_data:
        f.write("%s\n" % item)

from datasets import load_dataset
# Loading the dataset
text_datasets = {
    "train": ['/text_data/train.txt'],
    "eval": ['/text_data/val.txt'],
    "test": ['/text_data/test.txt']
}

dataset = load_dataset("text", data_files=text_datasets, cache_dir="/bpe_token")

from tokenizers import Tokenizer

# Load the tokenizer
loaded_tokenizer = Tokenizer.from_file('/space_bpe')

from tokenizers import Encoding
from typing import List

def pad_encodings(encodings: List[Encoding]):
    max_length = max([len(encoding.ids) for encoding in encodings])
    input_ids = []
    attention_mask = []
    for encoding in encodings:
        input_ids.append(encoding.ids + [0] * (max_length - len(encoding.ids)))
        attention_mask.append(encoding.attention_mask + [0] * (max_length - len(encoding.attention_mask)))
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def encode_batch(batch):
    encodings = loaded_tokenizer.encode_batch(batch["text"])
    return pad_encodings(encodings)

tokenized_dataset = dataset.map(
    encode_batch,
    batched=True,
    num_proc=4,  # Increase if your machine has more cores
    remove_columns=["text"],
)

# Save tokenized dataset
tokenized_dataset.save_to_disk("/tokenized_data")
