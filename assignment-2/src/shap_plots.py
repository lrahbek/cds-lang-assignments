import numpy as np
import sklearn
import shap
import os
from joblib import dump, load
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def article_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",
                        "-i", 
                        required = True,
                        type = int, 
                        help="The index of the article that a shap plot should be generated from")                           
    args = parser.parse_args()
    return args

def LRC_shapplot(ind, explainer, shap_values, X_test_array, feature_names, y_test_array):
    plot = shap.plots.force(explainer.expected_value, 
                            shap_values[ind,:], 
                            X_test_array[ind,:],
                            feature_names=feature_names, 
                            contribution_threshold = 0.05)
    outpath = os.path.join("out", "shap_plots", f"plotI{ind}.html")
    shap.save_html(outpath, plot)
    return print(f"The y label for the article is: {y_test_array[ind]}.\nThe .html plot has been saved to {outpath}")

def main():
    args = article_index()
    y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle(os.path.join("out", "features.pkl"))
    LRC = load(os.path.join("models", "LRC_accuracy_GS.joblib"))
    explainer = shap.LinearExplainer(LRC, X_train_features)
    shap_values = explainer.shap_values(X_test_features)
    X_test_array = X_test_features.toarray() 
    y_test_array = y_test.array
    LRC_shapplot(args.index, explainer, shap_values, X_test_array, feature_names, y_test_array)

if __name__ == "__main__":
    main()
