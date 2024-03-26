import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pickle
 
# Load the stored data 
X_train, X_test, y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('../out/features.pkl')

# Fit logistic regression classifier to training set, save it in 'models' folder, 
# and test it on test set. 
classifier_LR = LogisticRegression(random_state=42).fit(X_train_features, y_train)
dump(classifier_LR, "../models/classifier_LR.joblib")
y_pred_LR = classifier_LR.predict(X_test_features)


# Calculate evalutation metrics and save them as txt file in 'out'folder 
classifier_LR_metrics = metrics.classification_report(y_test, y_pred_LR)

filepath_metrics_LR = open(r'../out/classifier_LR_metrics.txt', 'w')
filepath_metrics_LR.write(classifier_LR_metrics)
filepath_metrics_LR.close()