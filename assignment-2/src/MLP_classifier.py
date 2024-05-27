import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
import pickle
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import argparse
import vectorizer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
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
    tracker.start_task("Load vectorised data (MLP)")
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
    param_grid = [
        {"activation": ["logistic", "relu"],            
         "tol": [0.00001, 0.0001, 0.001],               
         "hidden_layer_sizes": [(50,), (75,), (100,)]}]
    grid = GridSearchCV(MLPClassifier(solver = "adam",
                                      early_stopping = True,
                                      max_iter=1000, 
                                      random_state = 42, 
                                      verbose = True),
                        param_grid,                          
                        scoring = score,
                        cv=5, 
                        verbose = 1) 
    return grid

def MLP_fit(X_train_features, y_train, score, gridsearch, model_path, tracker):
    """
    The function fits the vectorised training data, to the parameters defined in grid_params, ans saves the best
    performing model in the 'out' folder. 
    """
    if gridsearch == "%GS":
        tracker.start_task("Fit MLP model")
        MLP = MLPClassifier(solver = "adam",
                            early_stopping = True,
                            max_iter=1000, 
                            random_state = 42, 
                            verbose = True).fit(X_train_features, y_train)
        params = {key: MLP.get_params()[key] for key in MLP.get_params().keys() & {'activation', 
                                                                                   'tol', 
                                                                                   'hidden_layer_sizes'}}
        eval_path = os.path.join("out", "MLP_%GS")
        tracker.stop_task()
    elif gridsearch == "GS":
        tracker.start_task("Fit MLP model with GS")
        grid = grid_params(score)
        grid = grid.fit(X_train_features, y_train)
        MLP = grid.best_estimator_
        params = grid.best_params_
        eval_path = os.path.join("out", f"MLP_GS_{score}")
        tracker.stop_task()
    dump([MLP, params, eval_path], model_path, compress = 1)


def MLP_evaluate(X_test_features, y_test, gridsearch, model_path, tracker):
    """ The function evaluates the MLP classifier and saves the classification report to the out folder """
    tracker.start_task("Evaluate MLP model")
    MLP, params, eval_path = load(model_path)
    y_pred_MLP = MLP.predict(X_test_features)
    if gridsearch == "%GS":
        class_report  = f"The parameters were set to the following:\n{params}n\nClassification Report:\n\n{classification_report(y_test, y_pred_MLP)}"
    elif gridsearch == "GS":
        class_report = f"The best performing parameters:\n{params}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_MLP)}\n\nMore info on the hyperparameters tuned etc. can be found in the README.md file"
    outpath_report = open(eval_path+"metrics.txt", 'w')
    outpath_report.write(class_report)
    outpath_report.close()
    tracker.stop_task()
    return print("Classification report for the MLP Classifier is saved to the out folder")

def plot_MLP_training(model_path, tracker):    
    """ Plot training loss and validation accuracy and save to out folder """
    tracker.start_task("Plot MLP training")
    MLP, params, eval_path = load(model_path)
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.plot(MLP.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("loss")
    plt.subplot(1,2,2)
    plt.plot(MLP.validation_scores_)
    plt.title("Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("accuracy")
    plt.savefig(eval_path+"plot.png")
    tracker.stop_task()

def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = get_arguments()
    model_path = os.path.join("models", f"MLP_{args.score}_{args.gridsearch}.joblib")
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data(tracker)
    MLP_fit(X_train_features, y_train, args.score, args.gridsearch, model_path, tracker)
    MLP_evaluate(X_test_features, y_test, args.gridsearch, model_path, tracker)
    plot_MLP_training(model_path, tracker)
    tracker.stop() 

if __name__ == "__main__":
    main()
