import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
import pickle
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# Load the stored data 
X_train, X_test, y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('../out/features.pkl')

# Fit MLP classifier to training set, save it in 'models' folder, 
# and test it on test set. 
classifier_MLP = MLPClassifier(activation = "logistic",
                               hidden_layer_sizes = (20,), 
                               max_iter=1000, 
                               random_state=42).fit(X_train_features, y_train)
dump(classifier_MLP, "../models/classifier_MLP.joblib")

y_pred_MLP = classifier_MLP.predict(X_test_features)


# Calculate evalutation metrics and save them as txt file in 'out'folder 
classifier_MLP_metrics = metrics.classification_report(y_test, y_pred_MLP)

filepath_metrics_MLP = open(r'../out/classifier_MLP_metrics.txt', 'w')
filepath_metrics_MLP.write(classifier_MLP_metrics)
filepath_metrics_MLP.close()