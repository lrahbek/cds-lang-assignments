import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--input_path",
                        "-i", 
                        required = False,
                        default = "in/fake_or_real_news.csv",
                        help="The path for the input dataset, if it isn't the default, the column names arguments should also be changed")               
    parser.add_argument(
                        "--text_column",
                        "-t", 
                        required = False,
                        default = "text",
                        help="The name for the column in the dataset with the text, the default is 'text' which is the column name in the default dataset")               
    parser.add_argument(
                        "--label_column",
                        "-l", 
                        required = False,
                        default = "label",
                        help="The name for the column in the dataset with the labels, the default is 'label' which is the column name in the default dataset")               
    parser.add_argument(
                        "--vectorizer_path",
                        "-v", 
                        required = False,
                        default = "models/tfidf_vectorizer",
                        help="The path were the defined vectorizer should be dumped and loaded from")               
    args = parser.parse_args()
    return args

def load_and_split(in_path, text_col, label_col):
    """
    The function takes the filepath for a given dataset, and the name of the column with the text and the name of
    the column with the classification labels. It returns the dataset split into test and train, 20% for test and
    80% for train.
    """
    data = pd.read_csv(in_path, index_col=0)
    X = data[text_col]
    y = data[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def def_save_vectorizer(vect_path):
    """
    The function takes the path where the vectorizer should be saved, defines a tfidf vectorizer, with set 
    parameters, as seen below, saves and returns the vectorizer.
    """
    vectorizer = TfidfVectorizer(ngram_range = (1,2), 
                                lowercase =  True, 
                                max_df = 0.95, 
                                min_df = 0.05, 
                                max_features = 500)
    dump(vectorizer, f"{vect_path}.joblib")
    return vectorizer

def fit_vectorizer(vect_path, X_train, X_test, y_train, y_test):
    """
    This function takes the path where the vectoizer is saved, and the split dataset, it fits and transforms the
    training data, transforms the test data and extracts the feature names. It saves the vectorised data, as well
    as the labels and feature names to the out folder in pickle format. 
    """
    vectorizer = load(f"{vect_path}.joblib")
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()     
    vectorized_data = [y_train, y_test, X_train_features, X_test_features, feature_names]
    f = open('out/features.pkl', 'wb' )
    pickle.dump(vectorized_data, f)
    f.close() 

def main():
    args = get_arguments()
    X_train, X_test, y_train, y_test = load_and_split(args.input_path, args.text_column, args.label_column)
    def_save_vectorizer(args.vectorizer_path)
    fit_vectorizer(args.vectorizer_path, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()