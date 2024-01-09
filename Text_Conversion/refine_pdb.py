import os
import sys
import glob
import pandas as pd

### version 231221

### operation : python refine_pdb.py [input_pdb_path] [receptor_id] [receptor_chain_id] [ligand_receptor_id] [ligand_id] [unique_job_id]
# example : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python refine_pdb.py /ssd002/hhjung/3bmGPT_hhjung_230921/input_pdb/5edq_A_rec_5edq_5n3_lig_crystal.pdb 5edq A 5edq 5n3 test

### read pdb file
input_pdb_path = sys.argv[1] 
receptor_id = sys.argv[2]
receptor_chain_id = sys.argv[3]
ligand_receptor_id = sys.argv[4]
ligand_id = sys.argv[5] 
unique_job_id = sys.argv[6]

ligand_id_c = ligand_id.upper()  # 5N3
ligand_id_l = ligand_id.lower()  # 5n3

work_dir = os.getcwd()

refined_pdb_dir = work_dir + "/refined_pdb"
if not os.path.isdir(refined_pdb_dir) :
    os.system("mkdir " + refined_pdb_dir)

general_pdb_name = receptor_id + "_" + receptor_chain_id + "_rec_" + ligand_receptor_id + "_" + ligand_id + "_lig_" + unique_job_id + "_general.pdb"
general_pdb_path = refined_pdb_dir + "/" + general_pdb_name

reduce_pdb_name = receptor_id + "_" + receptor_chain_id + "_rec_" + ligand_receptor_id + "_" + ligand_id + "_lig_" + unique_job_id + "_reduce.pdb"
reduce_pdb_path = refined_pdb_dir + "/" + reduce_pdb_name

refined_pdb_name = receptor_id + "_" + receptor_chain_id + "_rec_" + ligand_receptor_id + "_" + ligand_id + "_lig_" + unique_job_id + "_complex.pdb"
refined_pdb_path = refined_pdb_dir + "/" + refined_pdb_name

ligand_pdb_name = receptor_id + "_" + receptor_chain_id + "_rec_" + ligand_receptor_id + "_" + ligand_id + "_lig_" + unique_job_id + "_ligand.pdb"
ligand_pdb_path = refined_pdb_dir + "/" + ligand_pdb_name

# /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/5edq_A_rec_5edq_5n3_lig_refined.pdb

reduce_code_path = work_dir + "/reduce"

### get hetatm and atom
grep_command = "cat " + input_pdb_path + "|grep ^ATOM > " + general_pdb_path + ";echo TER >>" + general_pdb_path + ";cat " + input_pdb_path + "|grep ^HETATM >> " + general_pdb_path + ";echo END >>" + general_pdb_path + ";"
if os.path.isfile(input_pdb_path) :
    os.system(grep_command)

### remove hydrogen
reduce_cmd = reduce_code_path + " -trim " + general_pdb_path + ">" + reduce_pdb_path + ";"
if os.path.isfile(general_pdb_path) : 
    os.system(reduce_cmd)
print("### reduce pdb generated")
print(reduce_pdb_path)

### get matched chain data and arrange atom name
with open (reduce_pdb_path, "r") as reduce_pdb_file :
    lines = reduce_pdb_file.readlines()

atom_lines = []
hetatm_lines= []
for line in lines :
    line = line.strip()
    if line.startswith("ATOM") :
        if line[21] == receptor_chain_id :
            atom_lines.append(line)
    elif line.startswith("HETATM") :
        #print(line)
        if (line[17:20] == "UNL") or (line[17:20] == "LIG") or ((line[17:20] == ligand_id_c) and (line[21] == receptor_chain_id)) :
            hetatm_lines.append(line)

### if fail to get HETATM, return HETATM lines with ligand id "LIG"
def check_and_fix_hetatm(hetatm_lines, lines) :          ### version 231221
    new_hetatm_lines = []                                ### version 231221
    if len(hetatm_lines) == 0 :                          ### version 231221
        print("warning : invalid ligand data")           ### version 231221
        for line in lines :                              ### version 231221
            if line.startswith("HETATM") :               ### version 231221
                line = line[0:17] + "LIG" + line[20:]    ### version 231221
                new_hetatm_lines.append(line.strip())    ### version 231221
        return new_hetatm_lines                          ### version 231221
    else :                                               ### version 231221
        new_hetatm_lines = hetatm_lines                  ### version 231221
        return new_hetatm_lines                          ### version 231221

hetatm_lines = check_and_fix_hetatm(hetatm_lines, lines) ### version 231221
#print(hetatm_lines)                                     ### version 231221

def add_unique_identifiers(pdb_lines):
    atom_counts = {}
    updated_lines = []

    for line in pdb_lines:
        if line.startswith("HETATM") or line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            rest_of_line = line[16:]

            # Increment the atom count for the specific atom
            atom_counts[atom_name] = atom_counts.get(atom_name, 0) + 1

            # Format the atom name
            formatted_atom_name = atom_name.rjust(2)  # Right-align the atom name in 2 characters

            # Get the index and format it
            index = str(atom_counts[atom_name])

            # Concatenate atom name, index, and the rest of the line
            # Ensuring the atom name ends at column 13 and the index starts at column 14
            formatted_line = line[:12] + formatted_atom_name + index.ljust(2) + rest_of_line

            updated_lines.append(formatted_line)
        else:
            # If the line is not an atom/hetatm record, just add it as is
            updated_lines.append(line)

    return updated_lines

### check unique atom name
atom_name_list = []
#print(hetatm_lines)
for hetatm_line in hetatm_lines :
    atom_name = hetatm_line[12:16]
    #print(hetatm_lines) #
    #print(hetatm_lines[12:16])
    atom_name_list.append(atom_name)

#print(len(atom_name_list))
#print(len(set(atom_name_list)) )
if len(atom_name_list) > len(set(atom_name_list)) :
    hetatm_lines = add_unique_identifiers(hetatm_lines)
    print("converted_hetatm_lines")
    for line in hetatm_lines:
        print(line)
    refined_lines = [[line] for line in (atom_lines + ["TER"] + hetatm_lines + ["end"])]
else :
    refined_lines = [[line] for line in (atom_lines + ["TER"] + hetatm_lines + ["end"])]

### save as pdb file
refined_df = pd.DataFrame(refined_lines, columns = ["line"])
refined_df.to_csv(refined_pdb_path, index=False, header=None)
print("### refined pdb generated")
print(refined_pdb_path)

grep_hetatm_cmd = "cat " + refined_pdb_path + "|grep ^HETATM>" + ligand_pdb_path + ";echo END >>" + ligand_pdb_path + ";"
if os.path.isfile(refined_pdb_path) :
    os.system(grep_hetatm_cmd)
print(grep_hetatm_cmd)
print("### ligand pdb generated")
print(ligand_pdb_path)
