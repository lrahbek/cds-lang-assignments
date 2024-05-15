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
- Fits a MLP classifier to the training data, and saves the fitted model to the ```models``` folder, gridsearch is implemented, the parameters and hyperparameters used to tune the model are discussed in the Gridsearch section below. 
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder, and saves a plot of training loss and validation accuracy for the best performing model. 

### Gridsearch

All hyperparameters included in the gridsearch can be found at the [sckit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

For the Logistic Regression classifier ```solver```, ```penalty```, ```C``` and ```tol``` were tuned. Different penalties are available for different solvers, all possible combinations were investigated. 

- *solvers* = "lbfgs", "saga", "liblinear"
The solver determines the algorithm used when optimizing, three solvers were included for different reasons; 'lbfgs' is robust and the default solver for ```LogisticRegression```, 'liblinear' is recommended on smaller datasets, which the *Fake or Real News* dataset is, and 'saga' is overall well performing. 

- *penalties* = "l1", "l2", "None"
The penalties represent different regularization techinques, which helps balance between model fit and complexity. Not all solvers support all penalties, different penalties were included, as well as no penalty.  

- *C* = 1.0, 0.1, 0.01
The C hyperparameter defines the strength of the regulrazation on the model, smaller values regulates more and creates simpler models and bigger values allow for more complex models. The default is 1.0. 0.1 and 0.01 has also been included.

- *tol* = 0.00001, 0.0001, 0.001
Tolerance defines the threshold for when the model should stop training, the default is 0.0001. A smaller and a larger values was also included to introduce more range. 

For the MLP classifier ```activation```, ```hidden_layer_sizes``` and ```tol``` were tuned. The ```solver``` was kept at the default 'adam', and ```early_stopping``` included, which set 10% of the training data aside and validates continously, and stops when the validation accuracy is not improving by ```tol```for 10 epochs. The tolerance hyperparameter was set to the same three values used in the Logistic regression gridsearch.

- *activation* = "logistic", "relu"
The activation function is set to "relu", the "logistic" was included to evaluate wether the typical sigmoid activation function performed better. 

- *hidden_layer_sizes* = 50, 100, 150
The default is 100, 50 and 150 was included too, to evaluate wether more or less is necesary for the model to perform the best. 

## Data

The data used in this assignment is the *Fake or Real News* dataset, which can be downloaded [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The .csv file include 6335 articles, each represented in a row containing the title, the text, and a label indicating whether the article is fake or real. More info on the dataset can be found at the link above. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the ```fake_or_real_news.csv``` file from the source given above, and place it in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it.
- Open the virtual enviornment by writting ```source ./env/bin/activate``` in the terminal. 
- Preprocess the data by running ```python src/vectorizer.py``` in the terminal. (Alternatively, if the classifier scripts are run, and the vectorised data is not in the ```out``` folder, they will call the script and run it themselves). 
- To run the ```LR_classifier.py``` script, whether or not to implement gridsearch, and in the case of gridsearch, which evaluation metric should be tuned after should be passed in the terminal. An example on how to run the script with gridsearch and tuning for f1 score: 

```
python src/LR_classifier.py -g "GS" -s "f1"
```

- To run the ```MLP_classifier.py``` script, the metric that there should be tuned for should be specified (if anything other than accuracy). Following is an example of running the script and tuning for recall: 

```
python src/MLP_classifier.py -s "recall"
```

## Discussion 

The classification reports show ... 



*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*