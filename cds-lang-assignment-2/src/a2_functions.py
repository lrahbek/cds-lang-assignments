import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Function to save vectorised and additional data
def save_features(data, name, filepath):
    """
    The function saves the input data in the appropriate format. If the input data is not
    either an ndarray, csr_matrix or a Series, it will not save it. 
    """
    if type(data) == sp.sparse._csr.csr_matrix:
        sp.sparse.save_npz(
            os.path.join(filepath, f'{name}.npz'), 
            data, 
            compressed=True)
    elif type(data) == pd.core.series.Series:
        data.to_csv(os.path.join(filepath, f'{name}.csv'), index = False)
    elif type(data) == np.ndarray:
        data.dump(os.path.join(filepath, f'{name}.dat'))
    else: 
        print(f'{name}: not csr_matrix, ndarray or Series')

# Function to save model metrics
def save_metrics(model_metrics):
    """
    The function saves the input as a text file, with the name of the input variable. 
    """
    filename_metrics = get_var_name(model_metrics)
    filepath_metrics = open(f'../out/{filename_metrics}.txt', 'w')
    filepath_metrics.write(model_metrics)
    filepath_metrics.close()