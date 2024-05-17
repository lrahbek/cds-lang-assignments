import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from codecarbon import EmissionsTracker

def carbon_tracker(em_outpath):
    """ The function initalizes the carbon tracker """
    tracker = EmissionsTracker(project_name="Assignment-2",
                               output_dir=em_outpath)
    return tracker

def load_and_split(in_path, tracker):
    """ 
    The function takes the filepath for a dataset, the column containing the text data should be labeled 'text' and 
    the column containing the labels, should be labeled 'label'. It returns the dataset split into test and train, 
    20% for test and 80% for train.
    """
    tracker.start_task("Load data") 
    data = pd.read_csv(in_path, index_col=0)
    X = data["text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tracker.stop_task()
    return X_train, X_test, y_train, y_test

def define_vectorizer(vect_path, tracker):
    """ The function defines and saves a TFIDF vectoriser to a given path """
    tracker.start_task("Define vectorizer") 
    vectorizer = TfidfVectorizer(ngram_range = (1,2), 
                                lowercase =  True, 
                                max_df = 0.95, 
                                min_df = 0.05, 
                                max_features = 500)
    dump(vectorizer, f"{vect_path}.joblib")
    tracker.stop_task()

def fit_vectorizer(vect_path, X_train, X_test, y_train, y_test, tracker):
    """
    This function takes the path where the vectoizer is saved, and the split dataset, it fits and transforms the
    training data, transforms the test data and extracts the feature names. It saves the vectorised data, as well
    as the labels and feature names to the out folder in pickle format. 
    """
    tracker.start_task("Vectorize data")
    vectorizer = load(f"{vect_path}.joblib")
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()     
    vectorized_data = [y_train, y_test, X_train_features, X_test_features, feature_names]
    f = open('out/features.pkl', 'wb' )
    pickle.dump(vectorized_data, f)
    f.close() 

def main():
    tracker = carbon_tracker("../assignment-5/out")
    X_train, X_test, y_train, y_test = load_and_split("in/fake_or_real_news.csv", tracker)
    define_vectorizer("tfidf_vectorizer", tracker)
    fit_vectorizer("tfidf_vectorizer", X_train, X_test, y_train, y_test, tracker)
    tracker.stop() 

if __name__ == "__main__":
    main()