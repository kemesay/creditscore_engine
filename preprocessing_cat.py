import pandas as pd
import numpy as np
from config import config
'''
Preprocess the input data for the CatNN - Bucketize the numerical features
'''

class CatEncoder(object):

    """
        cat_col: list of column names of categorical data
        num_col: list of column names of numerical data
    """ 
    def __init__(self, cat_col, num_col):
        
        self.cat_col = cat_col
        self.num_col = num_col
        
        self.feature_columns = cat_col+num_col

        self.keep_values = {}
        self.num_bins = {}

    def fit_transform(self, X):

        print("Preprocess data for CatNN...")
        for idx in self.num_col:
            # Bucketize numeric features
            out, bins = pd.qcut(X.loc[:, idx], q=32, labels=False, retbins=True,   duplicates='drop')
            # print(out)
            X.loc[:, idx]= np.nan_to_num(out, nan=0).astype("int")
            self.num_bins[idx] = bins
            # print(self.num_bins[idx])

        X = X.astype("int")
        # Get feature sizes (number of different features in every column)
        feature_sizes = []

        for idx in self.feature_columns:
            feature_sizes.append(X.loc[:, idx].max()+1)
        return X, feature_sizes

    def transform(self, X):

        for idx in self.num_col:
            # Bucketize numeric features
            out = pd.cut(X.loc[:, idx], self.num_bins[idx], labels=False, include_lowest=True)
            X.loc[:, idx] = np.nan_to_num(out, nan=0).astype("int")
        return X
        