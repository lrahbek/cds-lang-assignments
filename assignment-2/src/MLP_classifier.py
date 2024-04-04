import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
import pickle
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--activation_function",
                        "-a", 
                        required = False,
                        default = "relu",
                        help="The activation function wanted for the MLP classifier, the default is relu. For more options see the scikitlearn documentation https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html")               
    args = parser.parse_args()
    return args

def load_data():
    """
    The function loads the pickled data from the out folder and returns, the training and test labels, 
    the feature names, and the training and test vectorised data. 
    """
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('out/features.pkl')
    return y_train, y_test, X_train_features, X_test_features, feature_names

def MLP_fit(X_train_features, y_train, activation_fun):
    """
    The function takes the vectorised training data and labels, as well as the activation function
    wanted, as a string. An MLP classifier is fitted to the data, with the given activation 
    funcition. It saves the classifier in the 'models' folder, where the activation function is 
    included in the file name, and returns the classifier as well. 
    """
    classifier_MLP = MLPClassifier( activation = activation_fun,
                                    hidden_layer_sizes = (100,), 
                                    max_iter=1000, 
                                    random_state=42, 
                                    verbose = True).fit(X_train_features, y_train)

    dump(classifier_MLP, f"models/classifier_MLP_{activation_fun}.joblib")
    return classifier_MLP

def MLP_evaluate(X_test_features, y_test, activation_fun):
    """
    The function takes the vectorised test data and labels, as well as the activation function used
    when fitting the classifier, as this denotes which MLP model is wanted from the models folder.
    It loads the MLP classifier and tests it on the test data, and saves the classification report 
    to the out folder. 
    """
    classifier_MLP = load(f"models/classifier_MLP_{activation_fun}.joblib")
    y_pred_MLP = classifier_MLP.predict(X_test_features)
    classifier_MLP_metrics = metrics.classification_report(y_test, y_pred_MLP)
    filepath_metrics_MLP = open(f'out/cl_MLP_metrics_{activation_fun}.txt', 'w')
    filepath_metrics_MLP.write(classifier_MLP_metrics)
    filepath_metrics_MLP.close()
    return print("Classification report for the MLP Classifier is saved to the out folder")

def main():
    args = get_arguments()
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data()
    MLP_fit(X_train_features, y_train, args.activation_function)
    MLP_evaluate(X_test_features, y_test, args.activation_function)

if __name__ == "__main__":
    main()