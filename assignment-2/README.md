# Assignment 2 - Text Classification Benchmarks

*Date: 07-03-2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 2 for Language Analytics. The objective of the assignment is to train benchmark machine learning classifiers on structured text data, using ```scikit-learn```, make and save understandable outputs and models, and save the results in clear ways. 

The code written for this assignment is seperated into three scripts that do the following: 
- vectorizer.py: 
    - The data is loaded and split into training anf test set. 
    - A vectorizer is defined and saved.
    - The data is vectorised and saved.
- LR_classifier.py: 
    - The vectorised data, and the labels are loaded 

- MLP_classifier.py: 

The output when running the code is two classifier models and a saved in ```models```, a classification report for each of the models and the vectorised data saved in ```out```.

## Data

The data used as a default in the classification tasks, is a dataset including 6335 articles, their titles, as well as a label indicating whether the article has been deemed *fake* or *real* news. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Run the bash script ```setup.sh``` from the command line.
- Run the bash script ```run.sh``` from the command line, which runs the three scripts in the ```src``` folder: 
    - vectorizer.py is run first, it vectorizes the input data and saves it in the ```out``` folder. It is possible to specify another dataset as input, for the script to run, it should have a column with the text that should be classified and a column with classification labels. The names of the labels should also be specified. Finally if the vectorizer should be saved somewhere else or with another name, this should also be specified.
    - LR_classifier.py is run second, and does not take any arguments. 
    - MLP_classifier.py is run third, and it should be specified if the activation function used when defining the classifier should be something other than 'relu'. Activation function options can be found in the [sckit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

    ```
    bash run.sh -i {"input path"} -t {"column name with text"} -l {"column name with labels"} -v {"vectorizer path"} -a {"the activation function used for the MLP"}
    ```

## Discussion 