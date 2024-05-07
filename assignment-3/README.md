# Assignment 3 - Query Expansion with Word Embeddings

*Date: 22/03/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 3 for Language Analytics. The objective of the assignment is to properly pre-process text data, use pre-trained word embeddings for query expansion, and to implement command line tools to generate results based on a given input. More specifically, the ```gensim``` pretrained wordembeddings model  ```glove-wiki-gigaword-50``` and the method ```model.most_similar()``` will be used to identify and extract the ten most similar words to a given keyword, these will then be used to perform the expanded query on the dataset, described in the Data section below. 

The ```keywordcounter.py``` script does the following: 
- Takes two arguments; an artist name and a keyword 
- Loads the *Spotify Million Song Dataset* and checks if any songs are by the artist given, if the artist is not in the dataset, 'Artist not found' is returned in the terminal. 
- Loads the ```glove-wiki-gigaword-50``` wordembedding model and returns a list of the ten most similar words, to the given keyword, and the keyword itself. 
- Cleans the texts by the given artists, by tokenising, making each token lower case and stripping punctuation. 
- Counts the number of texts any of the keywords are present in, by the given artist, and appends the results to the ```output.csv``` file. Each row in the file contains the keyword, the ten similar words, the name of the artist, the total number of songs by the artist, the total number of songs by the artist containing any of the words and finally the percentage of songs by the artist containing any of the words. 
- In the terminal the percentage of the artist's songs containg the words is returned. 

## Data

The data used in this assignment is the *Spotify Million Song Dataset*, a corpus containing 57,650 English-language songs with their titles, the artist, a link and the lyrics. The corpus can be found [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). 

## Usage and Reproducing of Analysis 

To perform the expanded query by running the script ```keywordcounter.py``` do the following: 
- Place the downloaded .csb file with the song data in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it.
- Run the bash script ```run.sh``` from the commandline, this opens the virtual environment and runs ```keywordcounter.py```, two arguments should be passed with it; the name of an artist and a given keyword. If the artist name is more than one word, it should be passed in quotation marks. 

  ```
  bash run.sh -a {'artist name'} -k {keyword of choice}
  ```
  
## Discussion 



*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*