# Assignment 3 - Query Expansion with Word Embeddings

*Date: 22/03/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 3 for Language Analytics. The objective of the assignment is to properly pre-process text data, use pre-trained word embeddings for query expansion, and to implement command line tools to generate results based on a given input. 

The code written returns the percentage of songs by a given artist that contain words related to the given query. The corpus used is described in the Data section. The expanded query is extracted using the gensim pretrained model ```glove-wiki-gigaword-50``` and the method ```model.most_similar()```, resulting in a list of 11 words, the ten closest words to the keyword, and the keyword. 
The output is printed in the terminal and appended to the .csv file in the ```out``` folder. 

## Data

The data used is a corpus containing 57,650 English-language songs with their titles, the artist, a link and the lyrics. The corpus can be found [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). When downloaded the .csv file shoule be placed in the ```in``` folder in the repository, to be able to run the code.

## Usage and Reproducing of Analysis 

To run the script ```keywordcounter.py``` do the following: 
- Place the .csv file with the song data in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line.
- Then run the bash script ```run.sh``` from the commandline, passing the two arguments of choice with it;

  ```
  bash run.sh -a {'artist name'} -k {keyword of choice}
  ```
  
## Discussion 


