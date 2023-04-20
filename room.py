import pandas as pd
import numpy as np
import config as config
import torch
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
            X[idx]= np.nan_to_num(out, nan=0).astype("int")
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
            X[idx] = np.nan_to_num(out, nan=0).astype("int")
        return X
        
    
########################################################################################################################
########################################################################################################################
class NumEncoder(object):
    """
        num_col: list of column names of numerical data
    """
    def __init__(self, cat_col, num_col, label_col):

        self.cat_col = cat_col
        self.num_col = num_col
        self.label_col = label_col

        self.means = []
        self.stds = []
        self.saved_sums = {}
        self.max_len = {}

    def fit_transform(self, df):

        print("Preprocess data for GBDT2NN...")

        # Preprocess the numerical data
        rows_num = self.min_max_scale(df.astype('float'))

        # Manual binary encoding
        if self.cat_col:
            rows_cat = self.binary_encoding(df)
            return np.concatenate([rows_num, rows_cat], axis=1)
        else:
            return rows_num

    def transform(self, df):
        # Preprocess the numerical data
        rows_num = self.min_max_scale(df.astype('float'), self.means, self.stds)

        # Manual binary encoding
        if self.cat_col:
            rows_cat = self.binary_encoding(df, self.max_len)
            return np.concatenate([rows_num, rows_cat], axis=1)
        else:
            return rows_num

    def refit_transform(self, df):

        # Update Means
        for item in self.num_col:
            self.saved_sums[item]['sum'] += df[item].sum()
            self.saved_sums[item]['cnt'] += df[item].shape[0]

        return self.transform(df)

    def binary_encoding(self, df, saved_bit_len=None):

        # print('Manual binary encode of categorical data')

        rows = []

        for item in self.cat_col:
            # print(item)
            # Get all values from column
            # feats = df[item].values.astype(np.uint8).reshape((-1, 1))
            feats = df.loc[:, item].astype(np.uint8)
            feats = np.array(feats)
            # print(feats)
            feats = np.reshape(feats, ((-1, 1)))
            # print(feats)

            # Compute the needed bit length based on the max size of the values

            if saved_bit_len is None:
                bit_len = len(bin(df.loc[:, item].astype(np.uint8).max())) - 2
                self.max_len[item] = bit_len
            else:
                bit_len = saved_bit_len[item]

            # change decimal to binary representation
            res = np.unpackbits(feats, axis=1, count=bit_len, bitorder='little')

            # append to all rows
            # rows = np.concatenate([rows,res],axis=1)
            rows.append(res)

        return np.concatenate(rows, axis=1)

    def min_max_scale(self, df, mean=None, std=None):
        # print('Min Max Scaling of numerical data')
        rows = df.loc[:, self.num_col] #.to_numpy()

        if mean is None:
            mean = np.mean(rows, axis=0)
            self.means = mean

        if std is None:
            std = np.std(rows, axis=0)
            self.stds = std

        rows = (rows - mean) / (std + 1e-5)
        return rows
########################################################################################################################################
########################################################################################################################################
def makePredictions(model, test_x, test_cat):
    device = config.config['device']
    test_x = torch.tensor(test_x, dtype=torch.float, device=device)
    test_cat = torch.tensor(test_cat, dtype= torch.float, device= device)
    testset = TensorDataset(test_x, test_cat)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.config['test_batch_size'])

    # Put model in evaluation mode to predict new value from new data
    model = model.to(device)
    model.eval()
    y_preds = [] 
    x_try = []
    with torch.no_grad():
        for data in testloader:
            inputs, inputs_cat = data
            inputs, inputs_cat = inputs.to(device), inputs_cat.to(device)
            outputs = model(inputs, inputs_cat)[0]
            print(outputs, 'ououtotutotutoutotuouotuotuouotott')
            max_value = 100
            min_value = 5
            x = ((outputs* (max_value - min_value)) + min_value)[4]
            # x = (outputs*100)[0]
            x_try.append(x.cpu().detach().numpy())
            y_preds.append(outputs.cpu().detach().numpy())
            my_array = np.array(x_try, dtype=np.float32)
            my_array = my_array.tolist()
            my_array  = ', '.join(str(num) for num in my_array)
            print('###############################################################################')
            print('')
            print("Dear Customer your credit score is" ,my_array, 'Thank you for working with US!')
            print('')
            print('')
            print('###############################################################################')
            print('')
    return np.concatenate(y_preds, axis=0)

###################################################################################################3
####################################################################################################
def preprocess(df, num_col, cat_col, label_col):
    print('\n################### Preprocess data ##################################\n')
    # Split the data into train and test data   
    train_df = df.sample(frac=config.config['rate'])
    test_df = df.drop(train_df.index)
    train_df = train_df.reset_index()
    train_m= train_df.drop('target', axis=1)
  
    # df1 = df1.drop('Customer_ID', axis=1)
    test_df = test_df.reset_index()
    # print("Is it up to this point")
    test_d= test_df.drop('target', axis=1)
    # Preprocess data for CatNN    
    ce = CatEncoder(cat_col, num_col)
    train_x_cat, feature_sizes = ce.fit_transform(train_m.copy())
    test_x_cat = ce.transform(test_d.copy()).astype('int32')
    # Preprocess data for GBDT2NN

    #train_dataset_without_target = df.drop(["target"],axis=1)
    # num_columns = [col for col in df.columns if col not in ["target"]]

    # X = df[num_columns]
    # y = df['target']
    ne = NumEncoder(cat_col, num_col, label_col)
    train_y = train_df['target']

    train_df= train_df.drop('target', axis=1)
    # print(train_df, 'train data with target and customer id')
    train_x = ne.fit_transform(train_df.copy())
    test_y = test_df['target']
    test_df = test_df.drop('target', axis=1)
    test_x= ne.transform(test_df.copy())


    return (train_x, train_y), (test_x, test_y), train_x_cat, test_x_cat, feature_sizes, ce, ne


def predict(model, data, ce, ne):
    data_cat = ce.transform(data.copy()).astype('int32')
    d_cat=   np.array(data_cat)
    data_cat = d_cat
    # print(data_cat[0])
    data_num = ne.transform(data.copy())
    data_num = np.concatenate([data_num, np.zeros((data_num.shape[0], 1), dtype=np.float32)], axis=-1)
    d_num= np.array(data_num)
    data_num = d_num
    # print(data_num[0])
    print('\n################### Make predictions #######################################\n')

    return makePredictions(model, data_num, data_cat)

##########################################################################################################################
##########################################################################################################################
# import argparse
import pandas as pd

num_col = ['2022-06-24','2022-06-25',	'2022-06-27',	'2022-06-28',	'2022-06-29',	'2022-06-30', '2022-07-01'	, '2022-07-02', '2022-07-04',
           '2022-07-05', '2022-07-06',	'2022-07-07',	'2022-07-08',	'2022-07-11',	'2022-07-12',	'2022-07-13',	'2022-07-14',	'2022-07-15',
           '2022-07-16','2022-07-18', '2022-07-19',	'2022-07-20',	'2022-07-21', '2022-07-22', '2022-07-23', '2022-07-25', '2022-07-26',	'2022-07-27',	'2022-07-28',	'2022-07-29',
           '2022-07-30',  '2022-08-01', '2022-08-02', '2022-08-03', '2022-08-04',	'2022-08-05', '2022-08-06','2022-08-08',	'2022-08-09', '2022-08-10', '2022-08-11',
           '2022-08-12', '2022-08-13', '2022-08-15', '2022-08-16', '2022-08-17',	'2022-08-18',	'2022-08-19', '2022-08-20', '2022-08-22', '2022-08-23',	'2022-08-24',	
           '2022-08-25', '2022-08-26', '2022-08-27', '2022-08-29', '2022-08-30', '2022-08-31', '2022-09-01', '2022-09-02',	'2022-09-03', '2022-09-05',	'2022-09-06', '2022-09-07',
           '2022-09-08', '2022-09-09','2022-09-10',	'2022-09-12', '2022-09-13', '2022-09-14', '2022-09-15',	 '2022-09-16',	'2022-09-17', '2022-09-19', '2022-09-20',
           '2022-09-21', '2022-09-22',	'2022-09-23',	'2022-09-24','AGE', 'Av_monthly_income']
cat_col = ['GENDER', 'Mar_status', 'Education_level', 'Occupation']
label_col = 'target'

def main():
   
    data = pd.read_csv('mondaylast_data.csv')
    # print(data)
####################################################################################################################################
####################################################################################################################################
    data = data.drop('CUSTOMER_ID', axis=1)
    
    data = data.loc[1:1705200, :]
    
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    # print(X_pca)
    # Apply K-means clustering to generate labels
    # kmeans = KMeans(n_clusters=2)
    kmeans = KMeans(n_clusters=2, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    
#     row_sums = data.sum(axis=1)
#     # assign the sum of rows to column
#     df1 = data.assign(row_sum=row_sums)
    

#     # max_value = df1['row_sum'].max()
#     # min_value = df1['row_sum'].min()
#     # print(max_value, min_value, 'your interval of these given one')


#    #Classification based on the given interval
#     bins = [-40.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 
#             500001.0, 600001.0, 700001.0, 1000000.0, 1000000001.0, 1297879501.0, 5000000001.0]

#     labels = [5 , 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,  80, 85, 90, 95, 100]
#     df1['row_sum'] = pd.cut(df1['row_sum'], bins=bins, labels=labels)
    
    data['labe'] = labels
    # print(df1.head(40))
     # data = data.drop('Customer_ID', axis=1)
    # df1 = df1.drop('target', axis=1)
    # # print(df1)
    # # data = df1.rename(columns={'row_sum': 'target'})
    # df1['row_sum'] = df1['row_sum'].astype('category')
    # df1['row_sum'] = df1['row_sum'].cat.codes
    # # create a sample dataset
    # max_value = df1['row_sum'].max()
    # min_value = df1['row_sum'].min()
    # df1['target'] = (df1['row_sum'].values - min_value) / (max_value - min_value)
    # # print(df1.head(40))
    # df1 = df1.drop('row_sum', axis=1)
    # data = df1
    
    data = data.drop('target', axis=1)
    data = data.rename(columns={'labe': 'target'})
#######################################################################################################################################
    dat = pd.read_csv('mondaylast_data.csv')
    dat = dat.drop('CUSTOMER_ID', axis=1)
    lab = 5
    dat = dat.iloc[[lab ]]
    dat.to_csv('indiv.csv', index=False)
    dt12 = pd.read_csv('indiv.csv')
    
    # print(dt12)
#######################################################################################################################################
# data = data.drop('Customer_ID', axis=1)
    dt = data.loc[3:11, :]
    # print(dt.head(9))
    dt = pd.concat([dt12, dt]).reset_index(drop=True)
    print(dt)

    train_num, test_num, train_cat, test_cat, feature_sizes, ce, ne = preprocess(data, num_col, cat_col, label_col)

    #(train_x, train_y), (test_x, test_y), train_x_cat, test_x_cat, feature_sizes, ce, ne

    the_model = torch.load("model12.pt")
    the_model.eval()
    predict(the_model, dt, ce, ne)

if __name__ == '__main__':
    main()