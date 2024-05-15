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
#import vectorizer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-2",
                               output_dir=em_outpath)
    return tracker

def get_arguments():
    parser = argparse.ArgumentParser()
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
    pipe = Pipeline([('MLP' , MLPClassifier(solver = "adam",
                                            early_stopping = True,
                                            max_iter=1000, 
                                            random_state = 42, 
                                            verbose = True))])
    param_grid = [
        {"MLP__activation": ["logistic", "relu"],            
         "MLP__tol": [0.00001, 0.0001, 0.001],               
         "MLP__hidden_layer_sizes": [(50,), (100,), (150,)]}]
    MLP = GridSearchCV(pipe,                                
                       param_grid,                          
                       scoring = score,
                       cv=2, 
                       verbose = 1) 
    return MLP

def MLP_fit(X_train_features, y_train, score):
    """
    The function fits the vectorised training data, to the parameters defined in grid_params, ans saves the best
    performing model in the 'out' folder. 
    """
    #tracker.start_task("Fit MLP model with GS")
    MLP = grid_params(score)
    MLP = MLP.fit(X_train_features, y_train)
    dump(MLP, f"models/MLP_{score}.joblib")
    #track.stop_task()

def MLP_evaluate(X_test_features, y_test, score):
    """ The function evaluates the MLP classifier and saves the classification report to the out folder """
    #tracker.start_task("Evaluate MLP model")
    MLP = load(f"models/MLP_{score}.joblib")
    y_pred_MLP = MLP.predict(X_test_features)
    class_report = f"The best performing parameters when tuning for {score}:\n{MLP.best_params_}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_MLP)}\n\nInfo on the hyperparameters tuned etc. can be found in the README.md file"
    outpath_report = open(f'out/MLP_metrics_{score}.txt', 'w')
    outpath_report.write(class_report)
    outpath_report.close()
    #tracker.stop_task()
    return print("Classification report for the MLP Classifier is saved to the out folder")

def plot_MLP_training(outpath, score):    
    """ Plot training loss and validation accuracy and save to out folder """
    #tracker.start_task("Plot MLP training")
    MLP = load(f"models/MLP_{score}.joblib")
    plt.figure(figsize = (12,6))
    plt.title("Training Plot for MLP Classifier")
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
    plt.savefig(outpath)
    #tracker.stop_task()

def main():
    args = get_arguments()
    #y_train, y_test, X_train_features, X_test_features, feature_names = load_data()
    #MLP_fit(X_train_features, y_train, args.score)
    #MLP_evaluate(X_test_features, y_test, args.score)
    plot_MLP_training(f"out/MLP_train_{args.score}.png", args.score)

if __name__ == "__main__":
    main()