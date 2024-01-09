from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


# Define the tokenizer
tokenizer = Tokenizer(models.BPE())

# Define the trainer
trainer = trainers.BpeTrainer(
    vocab_size=50257,
    show_progress=True,
    special_tokens=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"],
)

# Define pre_tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Here data_0718_re_2 must be a list of paths to your files.
# Also, save_bpe must be the path where you want to save the tokenizer
data_0718_re_2 = ['/mnt/prj/AJ/dock_bert/scripts/data_0718_re_2.json']
save_bpe = '/mnt/prj/AJ/dock_bert/scripts/0718_bpe'

# Train the tokenizer
tokenizer.train(files=data_0718_re_2, trainer=trainer)

# Save the tokenizer
tokenizer.save(save_bpe)