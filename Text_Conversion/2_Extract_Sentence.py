import os
import sys
import glob
import pandas as pd
import csv
import re

### read interaction output
inter_df = pd.read_csv("converted_interaction.txt", header=None)

### sentence generation and extraction
inter_df = inter_df[0:(len(inter_df)-1)]
combined = []
for line in inter_df[0]:
    parts = line.strip().split()
    if len(parts) >= 3:  # make sure there are enough columns
        new_string = parts[-4] + '-' + parts[-3]
        combined.append(new_string)

### save binding sentence
combined_str = ' '.join(combined)
with open("interaction_sentence.txt", "w") as f:
    f.write(combined_str)