# Assignment 2 - Text Classification Benchmarks

*Date: 07-03-2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 2 for Language Analytics. The objective of the assignment is to train benchmark machine learning classifiers on structured text data, using ```scikit-learn```, make and save understandable outputs and models, and save the results in clear ways. More specifically, a ```TfidfVectorizer``` will be used to vectorize and extract features from the *Fake or Real News* dataset, these features will be used in training two binary classification models to classify news articles as either 'REAL' or 'FAKE'. The ```LogisticRegression``` classifier and ```MLPClassifier``` will be used for this purpose, for both, gridsearch parameters will be set up to be able to identify the parameters that perform the best. Further, the evaluation metric the gridsearch should tune for can be passed as an argument, the default is accuracy. Three scripts were made for this assignment, each described below: 

The ```vectorizer.py``` script does the following: 
- Loads and splits the data into a test and train set. 
- Defines and saves a TFIDF vectorizer to the ```models``` folder. 
- Fits and vectorises the training data, and vectroizes the test data, then saves the extracted features to the ```out``` folder.

The ```LR_classifier.py``` script does the following: 
- Loads the vectorised features saved in the ```out``` folder. 
- Fits a logistic regression classifier to the training data, and saves the fitted model to the ```models``` folder. When running the script, it can be specified that gridsearch should be implemented, the parameters and hyperparameters used to tune the model are discussed in the Gridsearch section below.
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder. 

The ```MLP_classifier.py``` script does the following: 
- Loads the vectorised features saved in the ```out``` folder. 
- Fits a MLP classifier to the training data, and saves the fitted model to the ```models``` folder. Gridsearch can also be implemented here. The parameters and hyperparameters used to tune the model are discussed in the Gridsearch section below. 
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder. Additionally, it saves a plot of training loss and validation accuracy for the best performing model. 

### Gridsearch

All hyperparameters included in the gridsearch can be found at the sckit-learn documentation, for the [LogsiticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, and for the [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). The parameters used in tunning for both classifiers are explained below: 

**Logistic Regression Classifier**

For the Logistic Regression classifier ```solver```, ```penalty```, ```C``` and ```tol``` were tuned. Different penalties are available for different solvers, all combinations possible were included in the gridsearch. Additionally the parameters max_iter and random_state were set to 1000 and 42 (both for the base model and the model were gridsearch was implemented). 

|Parameter|Values|Explanation|
|---------|------|-----------|
|```solver```|"lbfgs", "saga", "liblinear"| Determines the optimization algorithm. The different solvers were included for different reasons; 'lbfgs' is robust and the default solver, 'liblinear' is recommended on smaller datasets, and 'saga' is overall well performing. |
|```penalty```|"l1", "l2", "None"|Determines the regularization technique implemented, helping to balance between model fit and complexity. Different penalties are available for different solvers, leading to choosing these three. |
|```C``` | 1.0, 0.1, 0.01|Determines the strength of regularization, the larger the value the less regulated the model is. The default is 1.0 |
| ```tol```|0.00001, 0.0001, 0.001|Determines the threshold for when the model should stop training, the default is 0.0001. |

**MLP Classifier**

For the MLP classifier ```activation```, ```hidden_layer_sizes``` and ```tol``` were tuned. For both the base model and the model where gridsearch was implemented, the ```solver``` was kept at the default 'adam', and ```early_stopping``` was set to True. This sets 10% of the training data aside and validates continously, and stops when the validation accuracy is not improving by ```tol```for 10 epochs. Further, max_iter and random_state was set to 1000 and 42. 

|Parameter|Values|Explanation|
|---------|------|-----------|
|```activation```|"logistic", "relu"|Determines the activation function in the nodes. The default is relu. |
|```hidden_layer_sizes```|50, 75, 100|Determines sizes of the hidden layers, the default is 100. |
| ```tol```|0.00001, 0.0001, 0.001|Determines the threshold for when the model should stop training, the default is 0.0001. |


## Data

The data used in this assignment is the *Fake or Real News* dataset, which can be downloaded [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The .csv file include 6335 articles, each represented in a row containing the title, the text, and a label indicating whether the article is fake or real. More info on the dataset can be found at the link above. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the ```fake_or_real_news.csv``` file from the source given above, and place it in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it.
- Open the virtual enviornment by writting ```source ./env/bin/activate``` in the terminal. 
- Preprocess the data by running ```python src/vectorizer.py``` in the terminal. (Alternatively, if the classifier scripts are run, and the vectorised data is not in the ```out``` folder, they will call the script and run it themselves). 
- For both ```LR_classifier.py``` and ```MLP_classifier.py```, it should be specified whether or not to perform gridsearch with the flag -g and it can be specified wich metric the gridsearch should be tuned for with the flag -s. The default is accuracy. E.g. running the logistic regression classifier with gridsearch and tunning for f1 should be passed like this: 

```
python src/LR_classifier.py -g "GS" -s "f1"
```

Simply running the neural network classifier with no gridsearch should be passed like this: 

```
python src/MLP_classifier.py
```

- Finally, to exit the virtual enviorment write ```deactivate``` in the terminal. 

## Discussion 

Before comparing the performance of the two different classifiers, I would like to compare the classification reports from the two runs of the ```LogisticRegression``` classifier; one with gridsearch and one with the default parameters from ```scikit-learn```. All classification reports can be found in the ```out``` folder, where recall, precision and f1 is also given. The default parameters lead to an accuracy of 0.89 and the parameters found via gridsearch lead to an accuracy of 0.88. The difference is very little, but indicates that the amount of time and power used to perform the gridsearch in this situtation might be unnecessary. Educated estimates or just the default parameters, seem to be more than enough to get a very well performing model. 

Similarly to the ```LogisticRegression``` classifier, the ```MLPclassiffier``` both with and without gridsearch performed almost identically, both with accuracies at .89 and similar f1, recall and precision scores. When inspecting the loss- and validation accuracy curves they again look very much alike. One difference, is that the model where gridsearch was implemented ran for fewer iterations, around 20 compared to the default models 35. 
*hidden layer sizes
tolerance *

*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*