# Binding Sentence Generation

This repository procides a brief guide for generating binding sentences from a PDB file

## Gneration Process

The process follows the numerical order of the filenames:

1. **Generate a preprocessed PDB file** by selecting a specific ligand and reformatting it - 1_PDB_Preprocessing,py
2. **Generate an interactioin result** by simulating 3D bindings from a PDB file - 2_Text_Conversion.exe
3. **Extract binding sentences** from the interaction result - 3_Extract_Sentences.py

## Usage

For the first step, edit `1_PDB_Preprocessing,py` to preprocess original PDB file

The second step runs via shell script and uses a preprocessed PDB file as input

```bash
./2_Text_Conversion.exe -l [input_PDB_path] > [converted_interaction.txt]
```

For the third step, edit `3_Extract_Sentences.py` to set the input path (e.g., `converted_interaction.txt`) directly in the code.
This will generate an output file like `interaction_sentence.txt`.
