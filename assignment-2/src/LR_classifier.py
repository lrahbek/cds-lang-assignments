import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pickle

def load_data():
    """
    The function loads the pickled data from the out folder and returns, the training and test labels, 
    the feature names, and the training and test vectorised data. 
    """
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('out/features.pkl')
    return y_train, y_test, X_train_features, X_test_features, feature_names

def LR_fit(X_train_features, y_train):
    """
    The function takes the vectorised training data and labels and fit a logistic regression classifier to it.
    It saves the classifier in the 'models' folder, and returns the classifier as well. 
    """
    classifier_LR = LogisticRegression(random_state=42).fit(X_train_features, y_train)
    dump(classifier_LR, "models/classifier_LR.joblib")
    return classifier_LR

def LR_evaluate(X_test_features, y_test):
    """
    The function takes the vectorised test data and labels. It loads the logistic regrssion classifier
    and tests it on the test data, and saves the classification report to the out folder. 
    """
    classifier_LR = load("models/classifier_LR.joblib")
    y_pred_LR = classifier_LR.predict(X_test_features)
    classifier_LR_metrics = metrics.classification_report(y_test, y_pred_LR)
    filepath_metrics_LR = open(r'out/classifier_LR_metrics.txt', 'w')
    filepath_metrics_LR.write(classifier_LR_metrics)
    filepath_metrics_LR.close()
    return print("Classification report for the Logistic Regression Classifier is saved to the out folder")

def main():
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data()
    LR_fit(X_train_features, y_train)
    LR_evaluate(X_test_features, y_test)

if __name__ == "__main__":
    main()