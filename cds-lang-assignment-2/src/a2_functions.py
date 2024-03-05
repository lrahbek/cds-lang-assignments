import os
import sys

import pandas as pd
import numpy as np
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def save_data(data, name, filepath):
    if type(data) == sp.sparse._csr.csr_matrix:
        sp.sparse.save_npz(
            os.path.join(filepath, f'{name}.npz'), 
            data, 
            compressed=True)
    elif type(data) == pd.core.series.Series:
        data.to_csv(os.path.join(filepath, f'{name}.csv'))
    elif type(data) == np.ndarray:
        data.dump(os.path.join(filepath, f'{name}.dat'))
    else: 
        print(f'{name}: not csr_matrix or Series')


def load_data(name, filepath, form):
    if form == 'npz':
        name = sp.sparse.load_npz(filepath)
    elif form == 'csv':
        name = pd.read_csv(filepath)
    elif form == 'dat':
        name = np.load(filepath, allow_pickle=True)
    else: 
        print(f'{name}: not csr_matrix or Series')