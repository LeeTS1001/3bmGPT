import os
import sys
import glob
import pandas as pd
import csv

import re

### version 231102

### operation : python extract_sentence.py [input_pdb_path] 
# example : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python extract_sentence.py /ssd002/hhjung/3bmGPT_hhjung_230921/input_pdb/5em8_A_rec_5em8_5q4_lig_it1_it2_tt_1_complex.pdb

### generate enva output from pdb
pdb_path = sys.argv[1] # /ssd002/hhjung/3bmGPT_hhjung_230921/input_pdb/5em8_A_rec_5em8_5q4_lig_it1_it2_tt_1_complex.pdb
pdb_name = pdb_path.split("/")[-1]
pdb_dir = ("/").join(pdb_path.split("/")[0:-1])

work_dir = os.getcwd()
enva_code_path = work_dir + "/enva4.5" 

enva_output_dir = work_dir + "/output_enva"
if not os.path.isdir(enva_output_dir) :
    os.system("mkdir " + enva_output_dir)
enva_output_path = enva_output_dir + "/" + pdb_name.replace(".pdb", ".out")

enva_cmd = "cd " + pdb_dir + ";" + enva_code_path + " -l " + pdb_name + " > " + enva_output_path
print(enva_cmd)
os.system(enva_cmd)

### read enva output file and get word list
enva_df = pd.read_csv(enva_output_path, header=None)
row_count = len(enva_df) -1
enva_df = enva_df[0:row_count]
#print(enva_df)

enva_df.columns = ["string"]
enva_df["word"] = enva_df["string"].apply(lambda x : x[72:104].strip().replace(" ", "-"))
word_list = enva_df["word"].tolist()
#print(word_list)

### check word if false then drop the word from list
def satisfies_condition(sequence) :
    pattern = r'(\d+)_(\d+)_[A-Z]:[A-Z]_[A-Z]_[A-Z]-[A-Z]_[A-Z0-9]{1,3}:[A-Z0-9]{1,3}_[A-Z0-9]{1,3}_[A-Z0-9]{1,3}$'
    words = sequence.split()
    #words = ["2_6_C:C_C_N-R_CD:CG_CD_NE"]
    for word in words:
        if not re.match(pattern, word):
            print(word)
            return False
    return True

fine_word_list = []
for word in word_list :
    pattern = r'(\d+)_(\d+)_[A-Z]:[A-Z]_[A-Z]_[A-Z]-[A-Z]_[A-Z0-9]{1,3}:[A-Z0-9]{1,3}_[A-Z0-9]{1,3}_[A-Z0-9]{1,3}$'
    if re.match(pattern, word) :
        fine_word_list.append(word)

sentence = (" ").join(fine_word_list)
#print(sentence)

output_sentence_dir = work_dir + "/output_sentence"
if not os.path.isdir(output_sentence_dir) :
    os.system("mkdir " + output_sentence_dir)
output_sentence_path = output_sentence_dir + "/" + pdb_name.replace(".pdb", "_sentence.txt")

with open (output_sentence_path, "w") as output_sentence_file :
    output_sentence_file.write(sentence + "\n")
print(output_sentence_path)

