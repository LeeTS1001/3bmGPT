import os
import sys
import csv
from rdkit import Chem

### version 231201

### opeartion : python gen_smi.py [ligand_pdb_path]

# example1-1 (invalid pdb for openbabel) : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/8ico_A_rec_8ico_azt_lig_DPOLB_HUMAN_10_complex.pdb
# example1-2 (1-1  with valid manual smiles) : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/8ico_A_rec_8ico_azt_lig_DPOLB_HUMAN_10_complex.pdb "CC1(C)OCC2=C(Nc3n[nH]c4c(Cl)cccc34)N=C(N=C12)C1C=NN=C1"

# example2-1 (valid pdb for openbabel but defective smiles recognition) : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/8ico_A_rec_8ico_azt_lig_DPOLB_HUMAN_10_ligand.pdb
# example2-2 (2-1  with valid manual smiles) : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/8ico_A_rec_8ico_azt_lig_DPOLB_HUMAN_10_ligand.pdb "CC1(C)OCC2=C(Nc3n[nH]c4c(Cl)cccc34)N=C(N=C12)C1C=NN=C1"

# example3-1 (valid pdb for openbabel and successful smiles recognition) cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/5edq_A_rec_5edq_5n3_lig_test_ligand.pdb
# example3-2 (3-1 with invalid manual smiles) : cd /ssd002/hhjung/3bmGPT_hhjung_230921 ; time python gen_smi_test.py /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/5edq_A_rec_5edq_5n3_lig_test_ligand.pdb "123456"

### obabel smiles generation
work_dir = os.getcwd()
ligand_pdb_path = sys.argv[1]
ligand_pdb_id = (".").join(ligand_pdb_path.split("/")[-1].split(".")[0:-1])
smi_dir = work_dir + "/ligand_smiles"

def smiles_writer(smiles_path, smiles) : ### 231201
    with open (smiles_path, "w") as smiles_file: ### 231201
        tsv_writer = csv.writer(smiles_file, delimiter="\t") ### 231201
        tsv_writer.writerows([[smiles, "manual"]]) ### 231201

def obsmi_generator(ligand_pdb_path, ligand_pdb_id, smi_dir) :
    work_dir = os.getcwd()
    if not os.path.isdir(smi_dir) :
        os.system("mkdir " + smi_dir)

    obsmi_path = smi_dir + "/" + ligand_pdb_id + "_obsmiles.smi"
    
    obabel_cmd = "obabel " + ligand_pdb_path + " -O " + obsmi_path + ";"
    # obabel /ssd002/hhjung/3bmGPT_hhjung_230921/refined_pdb/5edq_A_rec_5edq_5n3_lig_test_ligand.pdb -O outputfile.smiles
    print(obabel_cmd)
    os.system(obabel_cmd)

    print("### openbabel ligand smiles generated")
    print(obsmi_path)

    return obsmi_path

### rdkit smiles generation
def rdsmi_generator(obsmi_path) :
    with open (obsmi_path, "r") as obsmi_file :
        smi_reader = csv.reader(obsmi_file, delimiter="\t")
        obsmi = list(smi_reader)[0][0]

    #obsmi = "[C@H]12C(=[N]=C([C@@H]3C=NN=C3)N=C1C(C)(OC2)C)Nc1c2cccc(c2[nH]n1)Cl"
    rdmol = Chem.MolFromSmiles(obsmi)
    rdsmi = Chem.MolToSmiles(rdmol)

    rdsmi_path = smi_dir + "/" + ligand_pdb_id + "_rdsmiles.smi"
    with open (rdsmi_path, "w") as rdsmi_file :
        rdsmi_file.write(rdsmi + "\t\n")
    print("### rdkit ligand smiles generated")
    print(rdsmi_path)
    print("rdsmi : " + rdsmi)

### get smiles from sys.argv[2] 231201
obsmi_path = smi_dir + "/" + ligand_pdb_id + "_obsmiles.smi" ### 231201
rdsmi_path = smi_dir + "/" + ligand_pdb_id + "_rdsmiles.smi" ### 231201
try :
    obsmi_path = obsmi_generator(ligand_pdb_path, ligand_pdb_id, smi_dir)
    rdsmi_path = rdsmi_generator(obsmi_path)
except :
    print("smi generation failed")
    if len(sys.argv) > 2 : ### 231201
        manual_smiles = sys.argv[2] ### 231201
        rdmol = Chem.MolFromSmiles(manual_smiles)
        rdsmi = Chem.MolToSmiles(rdmol)
        smiles_writer(rdsmi_path, manual_smiles) ### 231201

print(obsmi_path) ### 231201
print(rdsmi_path) ### 231201
