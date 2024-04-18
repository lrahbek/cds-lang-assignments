# Assignment 4 - Emotion analysis with pretrained language models

*Date: 18/04/2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 4 for Language Analytics. The objective of the assignment is to use pretrained language models via HuggingFace, to extract meaningful structured information from unstructured text data, and to interpret and contextualize these results. More specifically, a finetuned model will be used to predict emotion scores for each line in the data in the 7 emotion categories; anger, disgust, fear, joy, neutral, sadness and surprise. The emotion category with the highest score will be extracted, and used to visually inspect the change in emotional profile of Game of Thrones across the eight seasons it ran. 

The model used can be found [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), and is a pretrained DistilRoBERTa-base model, finetuned on emotion-data. Details on the training data etc can be found at the link above. 

The code written for this assignment, does the following: 
- Loads in the ```j-hartmann/emotion-english-distilroberta-base``` classifier 
- Loads in the ```Game_of_Thrones_Scripts.csv``` dataset
- Runs the classifier on each sentence in the dataset and returns the emotion label with the highest score, as well as this labels score. 
- Saves a plot of the relative frequency of each of the seven emotions across the eight seasons 
    - One plot will be all emotion labels seperated into seasons 
    - The other plot will be all seasons seperated into emotion categories 

The script will save three files to the out folder, a .csv file with the complete dataset, including emotion labels and scores, and two plots visualising the emotions in the eight seasons. 

## Data 

The data used in this assignment can be found [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv). It is a .csv file containing every line in the scripts for the eight seasons of Game of Thrones. The .csv file also contains information on the season and episode the line was said, as well as which character said the line. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the dataset from the source described above, and place it in the in folder, the default filepath from the folder ```assignment-4``` to the data, is 'in/GoT-scripts/Game_of_Thrones_Script.csv
- Run the bash script ```setup.sh```
- Run the bash script ```run.sh```

It is possible to specify an altnative filepath when running the bash script by specifing it with -f {new path} after ```bash run.sh```

## Discussion

The visualization of the emotions in the eight seasons show;;

