# Assignment 3 - Query expansion with word embeddings

*Date: 22/03/2024*

Laura Givskov Rahbek 

## Description 

## Data

The data used is a corpus containing 57,650 English-language songs with their titles, the artist, a link and the lyrics. The corpus can be found [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). When downloaded the .csv file shoule be placed in the ```in``` folder in the repository, to be able to run the code.

## Usage and Reproducing of Analysis 

To run the code:
- Place the .csv file with the song data in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line.
- Then run the bash script ```run.sh``` from the commandline, passing the two arguments of choice with it;

  ```
  bash run.sh -a {'artist name'} -k {keyword of choice}
  ```

## Discussion 

______
## Description 

The objective of the assignment is to:

1. Pre-process texts in sensible ways;
2. Use pretrained word embeddings for query expansion;
3. Create a resusable command line tool for calculating results based on user inputs.

The code written returns the percentage of songs by a given artist that contain words related to the given query. The corpus used is described in the ```Data``` section. The expanded query is extracted using the gensim pretrained model ```glove-wiki-gigaword-50``` and the method ```model.most_similar()```, resulting in a list of 11 words, the ten closest words to the keyword, and the keyword. 
The output is printed in the terminal and appended to the .csv file in the ```out``` folder. 

