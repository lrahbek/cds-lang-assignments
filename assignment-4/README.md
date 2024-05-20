# Assignment 4 - Emotion analysis with pretrained language models

*Date: 18/04/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 4 for Language Analytics. The objective of the assignment is to use pretrained language models via HuggingFace, to extract meaningful structured information from unstructured text data, and to interpret and contextualize these results. More specifically, a finetuned model will be used to predict emotion scores for each line of text in the dataset (the dataset contains all lines spoken on Game of Thrones, for more see the Data section below), in the seven emotion categories; anger, disgust, fear, joy, neutral, sadness and surprise. The emotion category with the highest score will be extracted, and finally a plot will allow for visual inspection of the profile and development in the emotions of Game of Thrones across the eight seasons it ran. 

The model used can be found [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), and is a pretrained DistilRoBERTa-base model, finetuned on emotion-data. Details on the training data etc can be found at the link above. 

The ```GoT_emotions.py``` script does the following: 

- Loads in the ```j-hartmann/emotion-english-distilroberta-base``` classifier 
- Loads in the ```Game_of_Thrones_Scripts.csv``` dataset
- Runs the classifier on each sentence in the dataset and returns the emotion label with the highest score, as well as this labels score. 
- Saves a plot of the relative frequency of each of the seven emotions across the eight seasons 
    - One plot will be all emotion labels seperated into seasons 
    - The other plot will be all seasons seperated into emotion categories 

- The script will save three files to the out folder, a .csv file with the complete dataset, including emotion labels and scores, and two plots visualising the emotions in the eight seasons. 

## Data 

The data used in this assignment can be found [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv). It is a .csv file containing every line in the scripts for the eight seasons of Game of Thrones. The .csv file also contains information on the season and episode the line was said, as well as which character said the line. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the dataset from the source described above, and place it in the ```in``` folder, as ```GoT-scripts/Game_of_Thrones_Script.csv```
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it. 
- Run the bash script ```run.sh``` from the command line, this opens the virtual environment and runs the script in the ```src``` folder. 
-
- An alternative filepath can be specified using -f {filepath}. The script requires the input -n with either "w_neut" or "rm_neut", dictating whether to keep the neutral category or remove it, e.g. to run the analysis and including the neutral labels in the visualization: 

```
bash run.sh -n "w_neut"
```

## Discussion

The immidiate visuzalization of the emotional profile of Game of Thrones, shows a very large amount of neutral lines across all seasons. Above 40% of lines i all eight seasons had the highest score in the label neutral. A further analysis, would be able to shed light on whether many of these lines might have had another emotion label as prominent also, but as the neutral emotion is very similar across seasons, and obscures visual inspection of the other emotions, plots without neutral label was also made. 

When inspecting the plots, where the neutral emotion lavel have been removed, the seasons, in general, are very alike. First looking at the emotion prominence, distributed across emotion-label subplots; ```emotion_label_subplot_rm_neut.png```, the primary emotions across all seasons are anger and surprise. Anger i season 8 is the highest followed by season 1. The emotion with the most visible tendency is disugst, which falls steadily with season number. Sadness in season 1 and season 6 are slighlty higher than for the other seasons. Interestingly, the relative frequency of joy and fear are very alike, both around 5% across all seasons. Joy is at its highest in season 4 and fear is at its highest in season 7. When looking at the plot, in subplots representing seasons, ```Season_subplot_rm_neut.png```, the same tendencies are visible. 

A further inspection, could include an episode wise timeseries for each season or the scores for each emotion. 

*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*
