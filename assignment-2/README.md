# Assignment 2 - Text Classification Benchmarks

*Date: 07-03-2024*

Laura Givskov Rahbek 

## Description 

This folder contains assignment 2 for Language Analytics. The objective of the assignment is to train benchmark machine learning classifiers on structured text data, using ```scikit-learn```, make and save understandable outputs and models, and save the results in clear ways. More specifically, a ```TfidfVectorizer``` will be used to vectorize and extract features from the *Fake or Real News* dataset, these features will be used in training two binary classification models to classify news articles as either 'REAL' or 'FAKE'. The ```LogisticRegression``` classifier and ```MLPClassifier``` will be used for this purpose. For both classifiers ```GridSearchCV``` will be used to implement gridsearch and set up parameters and their values, to be able to identify the parameters that perform the best. When implementing gridsearch a five-fold cross validation is used, to increase robustness of the results. Further, the evaluation metric the gridsearch should tune for can be passed as an argument, the default is accuracy. Three scripts were made for this assignment, each described below: 

The ```vectorizer.py``` script does the following: 

- Loads and splits the data into a test and train set. 
- Defines and saves a TFIDF vectorizer to the ```models``` folder. 
- Fits and vectorises the training data, and vectroizes the test data, then saves the extracted features to the ```out``` folder.

The ```LR_classifier.py``` script does the following: 

- Loads the vectorised features saved in the ```out``` folder. 
- Fits a logistic regression classifier to the training data, and saves the fitted model to the ```models``` folder. It is possible to implement gridsearch when fitting the model, in both cases the ```max_iter``` = 1000 and ```random_state``` = 42, the remaining parameters are described below: 
  - If gridsearch is not implemented all paremeters are kept at their default values, available at the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
  - If gridsearch is implemented, ```solver```, ```penalty```, ```C``` and ```tol``` are tuned, the remaining parameters are kept at their default values.
      -  ```solver``` defines the optimization algorithm.'lbfgs', 'saga' and 'liblinear were included; 'lbfgs' is robust and the default solver, 'liblinear' is recommended on smaller datasets, and 'saga' is overall well performing. 
      - ```penalty``` determines the regularization technique implemented, helping to balance between model fit and complexity. Different penalties are available for different solvers, leading to choosing 'l1', 'l2' and None.  
      - ```C``` defines the strength of regularization, the larger the value the less regulated the model is. The default is 1.0, additionally 0.1 and 0.01 are included.
    - ```tol``` defines the threshold for when the model should stop training, the default is 0.0001, 0.00001 and 0.001 are included as well. 
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder, besides the classification report, the parameters used in the estimator can also be viewed in this file. 

The ```MLP_classifier.py``` script does the following: 

- Loads the vectorised features saved in the ```out``` folder. 
- Fits an MLP classifier to the training data, and saves the fitted model to the ```models``` folder. It is possible to implement gridsearch when fitting the model, in both cases the ```solver``` = 'adam', ```max_iter``` = 1000, ```random_state``` = 42 and ```early_stopping``` = True (the model uses 10% of the training data as validation, and the model stops training when the accuracy on the validation set does not increase by ```tol``` for 10 epochs), the remaining parameters are described below:
  - If gridsearch is not implemented the additional parameters are kept at their default values, available at the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
  - If gridsearch is implemented ```activation```, ```hidden_layer_sizes``` and ```tol``` are tuned, the remaining parameters are kept at their default values.
    -  ```activation``` defines how the nodes are activated, 'relu' and 'logistic' are included.
    -  ```hidden_layer_sizes``` determines the sizes of the hidden layers, the default is 100. To see if less can do it, 50, 75 and 100 was included as values in the gridsearch.
    -   ```tol``` defines the threshold for when the model should stop training, the default is 0.0001, 0.00001 and 0.001 are included as well. 
- Evaluates the performance of the model on the test data and saves the evaluation metrics to the ```out``` folder, besides the classification report, the parameters from the given estimator is also in this file. Additionally, it saves a plot of training loss and validation accuracy for the best performing model. 


## Data

The data used in this assignment is the *Fake or Real News* dataset, which can be downloaded [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The .csv file include 6335 articles, each represented in a row containing the title, the text, and a label indicating whether the article is fake or real. More info on the dataset can be found at the link above. 

## Usage and Reproducing of Analysis 

To reproduce the analysis: 
- Download the ```fake_or_real_news.csv``` file from the source given above, and place it in the ```in``` folder.
- Run the bash script ```setup.sh``` from the command line, it creates a virtual environment and installs packages and dependencies in to it.
- Open the virtual enviornment by writting ```source ./env/bin/activate``` in the terminal. 
- Preprocess the data by running ```python src/vectorizer.py``` in the terminal, (alternatively, if the classifier scripts are run, and the vectorised data is not in the ```out``` folder, they will call the script and run it themselves). 
- For both ```LR_classifier.py``` and ```MLP_classifier.py```, it should be specified whether or not to perform gridsearch with the flag -g (*gridsearch*) and it can be specified which metric the gridsearch should be tuned for with the flag -s (*score*), the default is accuracy.
  - E.g. running the logistic regression classifier with gridsearch and tunning for f1 should be written like this: 

    ```
    python src/LR_classifier.py -g "GS" -s "f1"
    ```

  - Or running the neural network classifier with no gridsearch should be written like this: 

    ```
    python src/MLP_classifier.py
    ```

- Finally, to exit the virtual environment write ```deactivate``` in the terminal. 

## Discussion 

Before comparing the performance of the two different classifiers, I would like to compare the classification reports from the two runs of the ```LogisticRegression``` classifier; one with gridsearch and one with the default parameters from ```scikit-learn```. All classification reports can be found in the ```out``` folder, where recall, precision and f1 is also given. The default parameters lead to an accuracy of 0.89 and the parameters found via gridsearch lead to an accuracy of 0.88. The difference is very little, but indicates that the amount of time and power used to perform the gridsearch in this situtation might be unnecessary. Educated estimates or just the default parameters, seem to be more than enough to get a very well performing model. 

Similarly to the ```LogisticRegression``` classifier, the ```MLPclassiffier``` both with and without gridsearch performed almost identically, both with accuracies at .89 and similar f1, recall and precision scores. When inspecting the loss- and validation accuracy curves they again look very much alike. One difference, is that the model where gridsearch was implemented ran for fewer iterations, around 20 compared to the default models 35. 
*hidden layer sizes
tolerance *

*```codecarbon``` was used to track the environmental impact when running this code, the results and an exploration of this can be found in the ```Assignment-5``` folder in the repository.*
