# Assignment 3 - Query expansion with word embeddings
**Language Analytics, Cultural Data Science**

*22/03/2024*

Laura Givskov Rahbek 


## Description 

The objective of the assignment is to:

1. Pre-process texts in sensible ways;
2. Use pretrained word embeddings for query expansion;
3. Create a resusable command line tool for calculating results based on user inputs.

The code written returns the percentage of songs by a given artist that contain words related to the given query. The corpus used is described in the ```Data``` section. The expanded query is extracted using the gensim pretrained model ```glove-wiki-gigaword-50``` and the method ```model.most_similar()```, resulting in a list of 11 words, the ten closest words to the keyword, and the keyword. 
The output is printed in the terminal and appended to the .csv file in the ```out``` folder. 


## Data

The data used can be found here: https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs, and is a corpus of 57,650 English-language songs. For each song, the artist, the title, a link and the lyrics are included. 


## Usage 

To use the code to find how many songs of a given artist contains words related to the keyword, do the following: 
- Run the bash script ``` setup.sh``` in the terminal (bash setup.sh)
- Run the bash script  ```run.sh``` in the terminal with the two arguments of choice (bash run.sh -a {arrtist}, -k {keyword})
