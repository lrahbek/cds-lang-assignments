# Assignment 1 - Extracting Linguistic Features Using spaCy

*Date: 23/02/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 1 for Language Analytics. The objective of the assignment is to work with input data arranged hierarchically in folders, to use ```spaCy``` to extract information from text data, and to save results and outputs in a clear and understandable way. More specifically the analysis aims to extract relevant textual features from the texts used (see Data section), and investigate whether these can capture the differences in the text. 

The ```feature_extraction.py``` script does the folllowing: 

- Loads an English language ```spaCy``` model, the default being ```en_core_web_md```. 
- Goes through each text in each of the subfolders in the ```in``` folder, for every text: 
  - The meta data is removed 
  - The relavtive frequency per 10,000 words of nouns, verbs, adjectives and adverbs are found 
  - The number og unique persons, locations and organisations are found
  - The word frquencies and named enteties are saved in a row in a .csv file, named according to the given subfolder.



The output when running the code is 14 .csv files, placed in the ```out``` folder; one for each subfolder in ```in```.

## Data

The data used is the *The Uppsala Student English Corpus (USE)*, which is a collection of essays in English written by Swedish university students. The data can be accesed at [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457), where additional information on the corpus can be found too. 

The subfolders reflect the term the essay was written in, subfolder starting with *a* was written in the first term, *b* in the second and *c* in the third. Further each subfolder in the three categories represent a specific essay prompt. 

## Usage and Reproducing of Analysis

To reproduce the analysis: 
- Download and unzip the ```USEcoprus``` folder, and place it in the ```in``` folder in the repository. 
- Run the bash script ```setup.sh``` from the command line.
- Run the bash script ```run.sh``` from the command line. To use another spaCy model than ```en_core_web_md```,, run the bash script with the name of one of the other english models (```en_core_web_sm``` or ```en_core_web_lg```):

  ```
  bash run.sh -m {'model'} 
  ```

## Discussion 

