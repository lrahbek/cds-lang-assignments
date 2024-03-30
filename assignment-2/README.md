# Assignment 2 - Text Classification Benchmarks

*Date: 07-03-2024*

Laura Givskov Rahbek 

## Description 

## Data

## Usage and Reproducing of Analysis 

## Discussion 



________
## Repository structure

The folder ```cds-lang-assignment-2``` contains four subfolders, with the solution to assignment 2 in language analytics: 

- ```in```: Contains a .csv file with texts from different newsarticles and a label indicating whether the given text is 'real' or 'fake'
- ```models```: Contains a TFIDF vectorizer, a fitted logistic regression classifier and a fitted MLP (neural network) classifier. 
- ```out```: Contains the evalutation metrics for both the logistic regression classifier and the MLP classifier, as well as a .pkl file with the vectorised data. 
- ```src```: Contains three scripts;
    - vectorizer.py: The data is loaded and split into test and train. The TFIDF vectorizer is defined and used to vectorize and fit the training data, and vectorize the test data, and the features are extracted.The vectroizer, vectorized data and features are saved. 
    - LR_classifier.py and MLP_classifier.py: In each script the saved data is retreived, the given classifier is defined and fit on the training set, and tested on the test set. The classifiers are saved, and the evaluation metrics for each are calculated and saved. 

## Description of assignment 2:

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different notebooks*. One script should train a logistic regression classifier on the data; the second notebook should train a neural network on the same dataset. Both notebooks should do the following:

- Save the classification report to a text file the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

### Objective

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

### Some notes

- Saving the classification report to a text file can be a little tricky. You will need to Google this part!
- You might want to challenge yourself to create a third script which vectorizes the data separately, and saves the new feature extracted dataset. That way, you only have to vectorize the data once in total, instead of once per script. Performance boost!

Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.

For this assignment, you are welcome to submit your code either as a Jupyter Notebook, or as ```.py``` script. If you do not know how to write ```.py``` scripts, don't worry - we're working towards that!

Lastly, you are welcome to edit this README file to contain whatever information you like. Remember - documentation is important.
