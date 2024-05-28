import numpy as np
import sklearn
import shap
import os
from joblib import dump, load
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-2",
                               output_dir=em_outpath)
    return tracker

def article_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",
                        "-i", 
                        required = True,
                        type = int, 
                        help="The index of the article that a shap plot should be generated from")   
    parser.add_argument("--LRCmodel",
                        "-l", 
                        required = True,
                        help="Which LogisticRegression model in the models folder that should be used, it should be given as the name of the file")   
    args = parser.parse_args()
    return args

def LRC_shapplot(ind, LRC, data_path, outpath):
    """ The shap library is used to save force plots of the given article """
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle(data_path)
    explainer = shap.LinearExplainer(LRC, X_train_features)
    shap_values = explainer.shap_values(X_test_features)
    X_test_array = X_test_features.toarray() 
    y_test_array = y_test.array
    plot = shap.plots.force(explainer.expected_value, 
                            shap_values[ind,:], 
                            X_test_array[ind,:],
                            feature_names=feature_names, 
                            contribution_threshold = 0.05)
    shap.save_html(outpath, plot)
    return print(f"The y label for the article is: {y_test_array[ind]}.\nThe .html plot has been saved to {outpath}")

def main():
    tracker = carbon_tracker(os.path.join("..","assignment-5", "out"))
    args = article_index()
    tracker.start_task("SHAP plot")
    outpath = os.path.join("out", "shap_plots", f"plotI{args.index}_{args.LRCmodel}.html")
    data_path = os.path.join("out", "features.pkl")
    LRC = load(os.path.join("models", args.LRCmodel+".joblib"))
    LRC_shapplot(args.index, LRC, data_path, outpath)
    tracker.stop_task()
    tracker.stop() 

if __name__ == "__main__":
    main()
