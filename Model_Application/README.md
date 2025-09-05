# Application on new binding data

This repository contains scripts for applying the pre-trained 3bmGPT model to new binding data. The primary functions are to generate sequence continuations and to calculate logit values from text-converted binding data.

## Prerequisites  

Input data : your binding data must be converted to a text format. Please refer to the Text_Conversion codes.
Pre-trained model : Download the 3bmGPT model files (model, config, and tokenizer) from Zenodo:
https://doi.org/10.5281/zenodo.17053575

## Usage

This repository provides two independent Python scripts for different tasks:
Word_Generation.py : Givin a text-converted binding sequences as input, this script generates a plausible continuation of that sequence.
Logit_Generation.py : For a given input inding sequence, this script calculates and outputs the corresponding logit values from 3bmGPT model.

