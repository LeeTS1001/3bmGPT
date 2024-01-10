# 3bmGPT (3D Binding Mode GPT)

Welcome to 3bmGPT github page

Disclaimer:
This repository is based on the following paper: "3bmGPT: A novel language model for transforming and analyzing protein-ligand binding interactions in drug discovery"
Please cite the original paper when using or developing this notebook.

# Files description
- Example_Input - Example data to apply to pre-trained 3bmGPT model
  - EGFR_5EDQ_SMILES.txt - SMILES of example ligand
  - EGFR_5EDQ_embedding_result.txt - Example embeddings generated from pre-trained 3bmGPT model
- GPT_Modelling
  - bpe_token_train.ipynb - To run BPE tokenizer
  - create_tokenized_data - To create tokenizer by trained data
  - data_preprocessing_tokenization.ipynb - To preprocess training data
  - dock_gpt2_train.py - To train GPT2 model
- Text_Converesion
  - enva4.5 - Excutable program to generate binding modes from refined PDB data
  - extract_sentence.py - To extract binding modes from interaction results, then converting them into text format
  - gen_smi_test.py - To generate ligand SMILES from PDB data
  - get_embedding_v2.py - To collect embedding vectors for PDB data applied to pre-trained 3bmGPT model
  - refine_pdb.py - To refine input PDB data
- 3bmGPT_Analysis_with_Pre_Trained.R - Running script from input embedding and ligand SMILES
- Pre_Trained_Data.zip
  - Meta_Data_From_Pre_Trained.csv - Meta data from analysis of pre-trained result
  - Normalized_Embedding_From_Pre_Trained.csv - Normalized embedding data from analysis of pre-trained result

# Usage
This manual is to perform 3bmGPT analysis with single binding interaction.
Pre-trained model was generated with 60 million interactions containing 18,450 protein-ligand complexes.
We conducted an in-depth analysis on 10,000 embedding data derived from pre-trained 3bmGPT model to annotate input interaction data.
The instructions provided entail generating final results for 3bmGPT using the embedded values generated from a specific protein-ligand interaction.

```shell
Rscript 3bmGPT_Analysis_with_Pre-Trained.R [normalized_embedding_path] [meta_data_path] [input_embedding_path] [input_smiles_path] [name]
```
Running Example
```shell
Rscript 3bmGPT_Analysis_with_Pre-Trained.R /Normalized_Embedding_From_Pre_trained.csv /Meta_Data_From_Pre_Trained.csv /EGFR_5EDQ_embedding_result.txt /EGFR_5EDQ_SMILES.txt EGFR_5EDQ
```
This script generates a list of the top 20 closely related interaction data based on the input interaction.

Fully functional web-based tool will be released along with comprehensive instructions.
