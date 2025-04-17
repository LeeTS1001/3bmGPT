# Binding Sentence Generation

This page provides a brief guide on using the code to generate binding sentences from a PDB file

## Gneration Process

The generation process follows the numerical order of the file names:

1. **Generate an interactioin result** by simulating 3D bindings from a PDB file - 1_Text_Conversion.exe
2. **Extract binding sentences** from the interaction result - 2_Extract_Sentences.py

## Usage

First step operates in shell script using a PDB file as input file

```shell
./1_Text_Conversion.exe -l [input_PDB_path] > [converted_interaction.txt]
```

Using the [converted_interaction.txt] output from first step, 2_Extract_Sentences.py will generate [interaction_sentence.txt]
