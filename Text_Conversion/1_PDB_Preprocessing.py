selected_ligand = 'ligand id' #The ligand ID, which is typically 3 letters in length
selected_chain = 'chain'

with open('Original PDB File', 'r') as infile, open('Input PDB', 'w') as outfile:
    write_mode = False
    for line in infile:
        if line.startswith('MASTER'):
            break
            
        if line.startswith('ATOM'):
            write_mode = True
        
        if write_mode and not line.startswith('CONECT'):
            if line.startswith('HETATM'):
                ligand_type = line[17:20].strip()
                chain_id = line[21].strip()
                
                if ligand_type == selected_ligand and chain_id == selected_chain:
                    outfile.write(line)
            else:
                outfile.write(line)