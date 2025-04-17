# Binding Sentence Generation

This repository procides a brief guide for generating binding sentences from a PDB file

## Gneration Process

The process follows the numerical order of the filenames:

1. **Generate an interactioin result** by simulating 3D bindings from a PDB file - 1_Text_Conversion.exe
2. **Extract binding sentences** from the interaction result - 2_Extract_Sentences.py

## Usage

The first step runs via shell script and uses a PDB file as input

```bash
./1_Text_Conversion.exe -l [input_PDB_path] > [converted_interaction.txt]
```

```bash
python 2_Extract_Sentences.py [converted_interaction.txt] > [interaction_sentence.txt]
```

For the second step, edit `2_Extract_Sentences.py` to set the input path (e.g., `converted_interaction.txt`) directly in the code.
This will generate an output file like `interaction_sentence.txt`.
