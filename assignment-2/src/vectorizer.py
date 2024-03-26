import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

# Read fake or real news data and define test and train split
filepath_data = os.path.join("..", "in","fake_or_real_news.csv")
news = pd.read_csv(filepath_data, index_col=0)
X = news["text"]
y = news["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,2), lowercase =  True, max_df = 0.95, min_df = 0.05, max_features = 500)    

# Fit vectorizer to data and extract feature names
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

# Save vectorizer in 'models' folder
dump(vectorizer, "../models/tfidf_vectorizer.joblib")

# Save vectorized data, extracted features etc
vectorized_data = [X_train, X_test, y_train, y_test, X_train_features, X_test_features, feature_names]

f = open('../out/features.pkl', 'wb' )
pickle.dump(vectorized_data, f)
f.close()