# GPT Model Training for Binding Sentences  

This page provides a brief guide on using the code to train the **3bmGPT** model.  

## Training Process  

The training process follows the numerical order of the script file names:  

1. **Create a BPE tokenizer** for binding sentences – `1_make_BPE_tokenizer.py`  
2. **Generate a tokenized dataset** from the original binding sentence dataset – `2_make_tokenized_dataset.py`  
3. **Configure and train the GPT model** using the tokenized dataset – `3_training_GPT.py`  

## Required Data  

The required binding data (`/dataset.json` in the code) is derived from the **CrossDocked2020** dataset, available at:  
🔗 [CrossDocked2020 Repository](https://github.com/gnina/models)  

## Step-by-Step Data Generation  

Each step produces the following output:  

1. **Filtered dataset** (removing sentences with a low word count) → `/filtered_data.json` (`1_make_BPE_tokenizer.py`)  
2. **BPE tokenizer file** with the specified vocabulary size → `/space_bpe` (`1_make_BPE_tokenizer.py`)  
3. **Train, validation, and test datasets** → `/text_data/train.txt`, `/text_data/val.txt`, `/text_data/test.txt` (`2_make_tokenized_dataset.py`)  
4. **Tokenized dataset** based on the train, validation, and test sets → `/tokenized_data` (`2_make_tokenized_dataset.py`)  
5. **Trained GPT model folder** → `/gpt2_model` (`3_training_GPT.py`)  
