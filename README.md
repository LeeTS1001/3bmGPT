# 3bmGPT (3D Binding Mode GPT)

Welcome to 3bmGPT github page.

This page provides codes for three parts of 3bmGPT: 1) generating input data from the 3D structure of protein-ligand interaction, 2) training the GPT model, and 3) applying newly introduced interaction data. The scripts for training the 3bmGPT model are available, but their direct application is hindered by the extensive dataset and the impracticality of establishing universal running settings. Instead of beginning the training process again, a web-based tool is available for utilizing a pre-trained model in binding interaction analysis. All the scripts for this tool are fully shared on this page.
For user convenience, a fully functional web-based tool is available at [3bmgpt.syntekabio.com](https://3bmgpt.syntekabio.com/).


Disclaimer:
This repository is based on the following paper: "3bmGPT: A novel language model for transforming and analyzing protein-ligand binding interactions in drug discovery"
Please cite the original paper when using or developing this notebook.

# Files description
- Example_Input - Example data to apply to pre-trained 3bmGPT model
  - EGFR_5EDQ_SMILES.txt - SMILES of example ligand
  - EGFR_5EDQ_logit_result.txt - Example logits generated from pre-trained 3bmGPT model
- GPT_Model_Training
  - 1_make_BPE_tokenizer.py - To run BPE tokenizer
  - 2_make_tokenized_dataset.py - To create tokenized data
  - 3_training_GPT.py - To train GPT model from tokenized data  
- Text_Converesion
  - 1_Text_Conversion.exe - To generate text-like interaction results from a PDB file
  - 2_Extract_Sentence.py - To extract binding modes from interaction results, then converting them into text format
- Model_Application
  - Logit_Generation.py - To generate logit values from input binding sentences
  - Word_Generation.py - To generate binding words from input binding sentences
- 3bmGPT_Analysis_with_Pre_Trained.R - Running script from input logits and ligand SMILES
- Pre_Trained_Data.zip
  - Meta_Data_From_Pre_Trained.csv - Meta data from analysis of pre-trained result
  - Normalized_logits_From_Pre_Trained.csv - Normalized logit data from analysis of pre-trained result

# Usage
This manual is to perform 3bmGPT analysis with single binding interaction.
Pre-trained model was generated with 60 million interactions containing 18,450 protein-ligand complexes.
We conducted an in-depth analysis on 10,000 logit data derived from pre-trained 3bmGPT model to annotate input interaction data.
The instructions provided entail generating final results for 3bmGPT using the logit values generated from a specific protein-ligand interaction.

```shell
Rscript 3bmGPT_Analysis_with_Pre-Trained.R [normalized_logit_path] [meta_data_path] [input_logit_path] [input_smiles_path] [name]
```
Running Example
```shell
Rscript 3bmGPT_Analysis_with_Pre-Trained.R /Normalized_logits_From_Pre_trained.csv /Meta_Data_From_Pre_Trained.csv /EGFR_5EDQ_logit_result.txt /EGFR_5EDQ_SMILES.txt EGFR_5EDQ
```
This script generates a list of the top 20 closely related interaction data based on the input interaction.
Comprehensive instructions will be released soon.


