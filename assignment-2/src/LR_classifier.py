import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
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
    features_path = os.path.join("out","features.pkl")
    if os.path.isfile(features_path) == False:
        main()
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle(features_path)
    tracker.stop_task()
    return y_train, y_test, X_train_features, X_test_features, feature_names

def grid_params(score):
    """
    The function defines parameters used in the gridsearch, and returns the gridsearch object which can then 
    be fitted. It takes the evaluation metric as an argument, the possible options can be found in the scikit-learn
    documentation. For info on the different hyperparameters, see the README.md file
    """
    C = [1.0, 0.1, 0.01]  
    tol = [0.00001, 0.0001, 0.001]
    param_grid = [
        {"C": C, "tol": tol, "solver": ["liblinear", "saga"] ,"penalty": ["l1", "l2"]},
        {"C": C, "tol": tol, "solver": ["lbfgs"] ,"penalty": [None, "l2"]}]
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state = 42),                                 
                       param_grid,                          
                       scoring = score,
                       cv=5, 
                       verbose = 1) 
    return grid

def LR_fit(X_train_features, y_train, gridsearch, score, model_path, tracker):
    """ 
    The function either fits the vectorised data to a default Logistic Regression model or performs a gridsearch
    on the parameters defined in grid_params() and fits to the best performing of these. If gridsearch is implemented
    it is possible to tune the parameters to another metric than the default 'accuracy'. The fitted model is saved
    to the 'models' folder. 
    """
    if gridsearch == "%GS":
        tracker.start_task("Fit LR model")
        LRC = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_features, y_train)
        params = {key: LRC.get_params()[key] for key in LRC.get_params().keys() & {'C', 'tol', 'solver', 'penalty'}}
        eval_path = os.path.join("out", "LR_%GS_rep.txt")
        tracker.stop_task()
    elif gridsearch == "GS":
        tracker.start_task("Fit LR model with GS")
        grid = grid_params(score)
        grid = grid.fit(X_train_features, y_train)
        LRC = grid.best_estimator_
        params = grid.best_params_
        eval_path = os.path.join("out", f"LR_GS_{score}_rep.txt")
        tracker.stop_task()
    dump([LRC, params, eval_path], model_path, compress = 1)


def LR_evaluate(X_test_features, y_test, gridsearch, model_path, tracker):
    """
    The function takes the vectorised test data and labels, and evaluates the given logistic regression classifier 
    on these. The classification report will be saved to the out folder, and named according to whether gridsearch
    was performed or not. 
    """
    tracker.start_task("Evaluate LR model")
    LRC, params, eval_path = load(model_path)
    y_pred_LR = LRC.predict(X_test_features)    
    if gridsearch == "%GS":
        class_report  = f"The parameters were set to the following values:\n{params}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_LR)}"
    elif gridsearch == "GS":
        class_report = f"The best performing parameters:\n{params}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_LR)}\n\nMore info on the hyperparameters tuned etc. can be found in the README.md file"
    outpath_report = open(eval_path, 'w')
    outpath_report.write(class_report)
    outpath_report.close()
    tracker.stop_task()
    return print("Classification report for the Logistic Regression Classifier is saved to the out folder")

def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    model_path = os.path.join("models", f"LRC_{args.score}_{args.gridsearch}.joblib")
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data(tracker)
    LR_fit(X_train_features, y_train, args.gridsearch, args.score, model_path, tracker)
    LR_evaluate(X_test_features, y_test, args.gridsearch, model_path, tracker)
    tracker.stop() 

if __name__ == "__main__":
    main()
