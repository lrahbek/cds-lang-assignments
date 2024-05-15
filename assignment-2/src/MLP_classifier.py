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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score",
                        "-s", 
                        required = False,
                        default = "accuracy",
                        help="Which metric the gridsearch should tune for")              
    args = parser.parse_args()
    return args

def load_data():
    """ Load the vectorised data, if the data has not been vectorised the vectorizer.py script will be run """
    if os.path.isfile('out/features.pkl') == False:
        main()
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('out/features.pkl')
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
                       cv=10, 
                       verbose = 1) 
    return MLP

def MLP_fit(X_train_features, y_train, score):
    """
    The function fits the vectorised training data, to the parameters defined in grid_params, ans saves the best
    performing model in the 'out' folder. 
    """
    MLP = grid_params(score)
    MLP = MLP.fit(X_train_features, y_train)
    dump(MLP, f"models/MLP_{score}.joblib")

def MLP_evaluate(X_test_features, y_test, score):
    """ The function evaluates the MLP classifier and saves the classification report to the out folder """
    MLP = load(f"models/MLP_{score}.joblib")
    y_pred_MLP = MLP.predict(X_test_features)
    class_report = f"The best performing parameters when tuning for {score}:\n{MLP.best_params_}\n\nClassification Report:\n\n{classification_report(y_test, y_pred_MLP)}\n\nInfo on the hyperparameters tuned etc. can be found in the README.md file"
    outpath_report = open(f'out/MLP_metrics_{score}.txt', 'w')
    outpath_report.write(class_report)
    outpath_report.close()
    return print("Classification report for the MLP Classifier is saved to the out folder")

def plot_MLP_training(outpath):    
    """ """
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

plot_MLP_training("out/MLP_training.png")

def main():
    args = get_arguments()
    y_train, y_test, X_train_features, X_test_features, feature_names = load_data()
    MLP_fit(X_train_features, y_train, args.score)
    MLP_evaluate(X_test_features, y_test, args.score)

if __name__ == "__main__":
    main()