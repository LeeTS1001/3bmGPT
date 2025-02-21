from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Open the dataset for reading
with open('/dataset.json', 'r') as f:
    # Load the JSON data from the file
    data = json.load(f)

# Word count definition
def count_words(s):
    return len(s.split())

# Set word count threshold
word_count_threshold = 20

# Generate filtered data by word count threshold
filtered_data = [elem for elem in unique_data if count_words(elem) >= word_count_threshold]

# Save the filtered data to a JSON file
with open('/filtered_data.json', 'w') as json_file:
    json.dump(filtered_data, json_file)

from multiprocessing import Pool

# Function to read and preprocess data
def read_and_preprocess(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Define the tokenizer and trainer as functions to be used in multiprocessing
def create_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer

# Set parameters for the trainer
def create_trainer():
    trainer = trainers.BpeTrainer(
        vocab_size=50257,
        show_progress=True,
        special_tokens=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"],
    )
    return trainer

# Train the tokenizer on combined data
def train_tokenizer(data, save_path):
    tokenizer = create_tokenizer()
    trainer = create_trainer()
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save(save_path)

# Set file directories
data_files = ['/filtered_data.json']
save_bpe = '/space_bpe'

# Use multiprocessing to read and preprocess data in parallel
with Pool(os.cpu_count()) as pool:
    data_chunks = pool.map(read_and_preprocess, data_files)

# Combine the preprocessed data
combined_data = []
for chunk in data_chunks:
    combined_data.extend(chunk)

# Train the tokenizer on the combined data
train_tokenizer(combined_data, save_bpe)