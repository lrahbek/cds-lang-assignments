import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
import argparse
import vectorizer
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-2",
                               output_dir=em_outpath)
    return tracker

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsearch",
                        "-g", 
                        required = True,
                        choices = ["GS", "%GS"],
                        help="Whether or not to perform gridsearch when fitting the model")               
    parser.add_argument("--score",
                        "-s", 
                        required = False,
                        default = "accuracy",
                        help="Which metric the gridsearch should tune for")               
    args = parser.parse_args()
    return args

def load_data(tracker):
    """ Load the vectorised data, if the data has not been vectorised the vectorizer.py script will be run """
    tracker.start_task("Load vectorised data (LR)")
    if os.path.isfile('out/features.pkl') == False:
        main()
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('out/features.pkl')
    tracker.stop_task()
    return y_train, y_test, X_train_features, X_test_features, feature_names

def grid_params(score):
    """
    The function defines parameters used in the gridsearch, and returns the gridsearch object which can then 
    be fitted. It takes the evaluation metric as an argument, the possible options can be found in the scikit-learn
    documentation. For info on the different hyperparameters, see the README.md file
    """
    pipe = Pipeline([('LRC' , LogisticRegression(max_iter=1000, random_state = 42))])
    C = [1.0, 0.1, 0.01]  
    tol = [0.00001, 0.0001, 0.001]
    param_grid = [
        {"LRC__C": C, "LRC__tol": tol, "LRC__solver": ["liblinear", "saga"] ,"LRC__penalty": ["l1", "l2"]},
        {"LRC__C": C, "LRC__tol": tol, "LRC__solver": ["lbfgs"] ,"LRC__penalty": [None, "l2"]}]
    LRC = GridSearchCV(pipe,                                
                       param_grid,                          
                       scoring = score,
                       cv=10, 
                       verbose = 1) 
    return LRC

def LR_fit(X_train_features, y_train, gridsearch, score, tracker):
    """ 
    The function either fits the vectorised data to a default Logistic Regression model or performs a gridsearch
    on the parameters defined in grid_params() and fits to the best performing of these. If gridsearch is implemented
    it is possible to tune the parameters to another metric than the default 'accuracy'. The fitted model is saved
    to the 'models' folder. 
    """
    if gridsearch == "%GS":
        tracker.start_task("Fit LR model")
        LRC = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_features, y_train)
        tracker.stop_task()
    elif gridsearch == "GS":
        tracker.start_task("Fit LR model with GS")
        LRC = grid_params(score)
        LRC = LRC.fit(X_train_features, y_train)
        tracker.stop_task()
    dump(LRC, f"models/LRC_{gridsearch}.joblib")


def LR_evaluate(X_test_features, y_test, gridsearch, score, tracker):
    """
    The function takes the vectorised test data and labels, and evaluates the given logistic regression classifier 
    on these. The classification report will be saved to the out folder, and named according to whether gridsearch
    was performed or not. 
    """
    tracker.start_task("Evaluate LR model")
    LRC = load(f"models/LRC_{gridsearch}.joblib")
    y_pred_LR = LRC.predict(X_test_features)
    if gridsearch == "%GS":
        class_report  = f"Classification Report:\n\n{classification_report(y_test, y_pred_LR)}\n\nThe parameters are all the default parameters define by scikit-learn, except the max_iter was set to 1000 and random_state to 42"
    elif gridsearch == "GS":
        class_report = f"The best performing parameters when tuning for {score}:\n{LRC.best_params_}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_LR)}\n\nInfo on the hyperparameters tuned etc. can be found in the README.md file"
    outpath_report = open(f'out/LRC_metrics_{gridsearch}.txt', 'w')
    outpath_report.write(class_report)
    outpath_report.close()
    tracker.stop_task()
    return print("Classification report for the Logistic Regression Classifier is saved to the out folder")

def main():
    tracker = carbon_tracker("../assignment-5/out")
    args = get_arguments()
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data(tracker)
    LR_fit(X_train_features, y_train, args.gridsearch, args.score, tracker)
    LR_evaluate(X_test_features, y_test, args.gridsearch, args.score, tracker)
    tracker.stop() 

if __name__ == "__main__":
    main()