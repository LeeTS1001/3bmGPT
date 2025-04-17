from glob import glob
import os
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
import numpy as np

# load tokenize dataset
tokenizer = PreTrainedTokenizerFast(tokenizer_file="/tokenizer_file")
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Load the model configuration
config = GPT2Config.from_pretrained('/checkpoint/config.json')
model = GPT2LMHeadModel(config)

# Load the state dict
state_dict = torch.load(
    '/checkpoint/pytorch_model.bin',
    map_location=torch.device('cpu')
)
model.load_state_dict(state_dict, strict=False)

#Generation example for a sample sentence
samp_sentence = '2_4_C:C_C_C-F_CD2:CE2_CD2_CG 1_4_C:C_C_C-D_OD2:OD1_OD2_CB 1_4_C:C_C_C-D_OD1:CB_OD1_OD2 1_6_C:C_C_C-Y_OH:CE1_OH_CE2 1_4_C:C_C_C-Y_OH:CE1_OH_CE2 2_6_C:C_C_C-F_CZ:CE1_CZ_CE2 2_4_C:C_C_C-F_CE2:CZ_CE2_CD2 1_4_C:C_C_C-L_CD2:CD1_CD2_CB 1_4_N:C_N_C-L_O:C_N_CA 1_6_C:C_C_C-L_O:C_N_CA 1_4_C:N_C_C-V_CG1:CA_CG1_CG2 1_6_C:C_C_C-V_CG2:CG1_CG2_CA 2_6_N:C_N_C-L_O:C_N_CA 2_6_O:N_O_C-M_CG:CB_CG_SD 1_4_C:C_C_C-M_SD:CG_SD_SD 2_6_C:C_C_C-M_CE:CG_SD_CG 1_3_C:N_C_C-L_CD1:CB_CD1_CD2 2_6_C:C_C_C-Y_OH:CE1_OH_CE2 1_4_C:C_C_C-V_CG1:CA_CG1_CG2 1_4_C:C_C_C-V_CG2:CG1_CG2_CA 2_6_C:C_C_C-W_CZ2:CH2_CZ2_CE2'

inputs = tokenizer(samp_sentence, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
outputs.logits.mean(dim=1).detach().numpy()



