# Assignment 2 - Text Classification Benchmarks

*Date: 07-03-2024*

Laura Givskov Rahbek 



For this exercise, you should write two different notebooks. One should train a logistic regression classifier on the data; the second notebook should train a neural network on the same dataset. Both notebooks should do the following:

Save the classification report to a text file the folder called out
Save the trained models and vectorizers to the folder called models

## Description 

This folder contains assignment 2 for Language Analytics. The objective of the assignment is to train benchmark machine learning classifiers on structured text data, using ```scikit-learn```, make and save understandable outputs and models, and save the results in clear ways. More specifically two different binary classification models will be trained on the *Fake or Real News* dataset, a logistic regression classifier and a neural network. Three scripts were made for this assignment, each described below: 

The ```vectorizer.py``` script does the following: 
- Loads and splits the data into a test and train set. 
- Defines and saves a TFIDF vectorizer to the ```models``` folder. 
- Fits and vectorises the training data, and vectroizes the test data, then saves the extracted features to the ```out``` folder.

The ```LR_classifier.py``` script does the following: 
- Loads the vectorised features saved in the ```out``` folder. 
- Fits a logistic regression classifier to the training data, and saves the fitted model to the ```models``` folder. 
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder. 

The ```MLP_classifier.py``` script does the following: 
- 

- Different options for the gridsearch can be found in the [sckit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

## Data

The data used in this assignment is the *Fake or Real News* dataset, which can be downloaded [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The .csv file include 6335 articles, each represented in a row containing the title, the text, and a label indicating whether the article is fake or real. More info on the dataset can be found at the link above. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the ```fake_or_real_news.csv``` file from the source given above, and place it in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line, it creates a virtual enviorment and installs packages and dependencies in to it.
- Run the bash script ```run.sh``` from the command line, this opens the virtual enviornment and runs all three scripts in ```src``` folder. 
    - ```vectorizer.py``` can take some arguments, written out in the table below (the default arguments fit the dataset *fake_or_real_news* if it is correctly placed in the ```in``` folder). 
    - ```MLP_classifier.py``` can also take some arguments, specfying the gridsearch, these are also written out in the table below. 

|Flag | Description                                                  | 
|-----|--------------------------------------------------------------|
|-i   | The path where the dataset is stored                         |
|-t   | Name of the column containing text that should be vectorised |
|-l   | Name of the column containing binary classification labels   |
|-v   | The path were the vectorizer should be saved to              |


## Discussion 

The classification reports show ... 