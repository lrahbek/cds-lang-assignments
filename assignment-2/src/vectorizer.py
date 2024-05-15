import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle


def load_and_split(in_path):
    """ 
    The function takes the filepath for a dataset, the column containing the text data should be labeled 'text' and 
    the column containing the labels, should be labeled 'label'. It returns the dataset split into test and train, 
    20% for test and 80% for train.
    """
    data = pd.read_csv(in_path, index_col=0)
    X = data[text_col]
    y = data[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def define_vectorizer(vect_path):
    """ The function defines and saves a TFIDF vectoriser to a given path """
    vectorizer = TfidfVectorizer(ngram_range = (1,2), 
                                lowercase =  True, 
                                max_df = 0.95, 
                                min_df = 0.05, 
                                max_features = 500)
    dump(vectorizer, f"{vect_path}.joblib")

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
    X_train, X_test, y_train, y_test = load_and_split("in/fake_or_real_news.csv")
    define_vectorizer("tfidf_vectorizer")
    fit_vectorizer("tfidf_vectorizer", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()