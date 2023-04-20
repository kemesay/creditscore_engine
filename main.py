import numpy as np
from helpers import AdamW
from DeepGBM import DeepGBM
from EmbeddingModel import EmbeddingModel
from preprocessing_cat import CatEncoder
from preprocessing_num import NumEncoder
from torch.utils.data import TensorDataset
from gbdt import TrainGBDT, SubGBDTLeaf_cls, get_infos_from_gbms
from helpers import outputFromEmbeddingModel, eval_metrics, printMetric
import torch
from trainModel import trainModel, evaluateModel, makePredictions
import config as config
# import pickle

'''
    Preprocess the data
    
        for GBDT2NN: Fill NaNs, Ordinal encode and then binary encode categoric features, Scale numeric features
        for CatNN: Fill NaNs, Bucketize numeric features, Filter categoric features, Ordinal encode all features
    
    Parameters:
        - df: dataframe with all trainings data
        - num_col: list of all numerical columns
        - cat_col: list of all categorical column
        - label_col: name of label column
        
    Returns:
        - train_num: data and label preprocessed for training
        - test_num: data and label preprocessed for testing
        - train_cat: data preprocessed for training CatNN
        - test_cat: data preprocessed for testing CatNN
        - feature sizes: list containing the number of different features in every column
'''


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


'''
    Train the model
        1. Train a GBDT and distill the knowledge from its leafs
        2. Train a Embedding Model using the leaf predictions as data
        3. Train a DeepGBM model (consisting of a GBDT2NN and a CatNN) using all data and the outputs from the Embedding Model
    Parameters:
        - train_num: data and label preprocessed for training
        - test_num: data and label preprocessed for testing
        - train_cat: data preprocessed for training CatNN
        - test_cat: data preprocessed for testing CatNN
        - feature sizes: list containing the number of different features in every col
    Returns:
        - deepgbm_model: trained model
        - optimizer: optimizer used during training
'''


def train(train_num, test_num, train_cat, test_cat, feature_sizes, save_model=False):
    x_train, y_train = train_num
    x_test, y_test = test_num

    print('\n################### Train model #######################################\n')

    print("\nTrain GBDT and distill knowledge from it...")
    # Train GBDT and distill knowledge from it
    gbm = TrainGBDT(x_train, y_train, x_test, y_test)
    gbms = SubGBDTLeaf_cls(x_train, x_test, gbm)
    used_features, tree_outputs, leaf_preds, test_leaf_preds, group_average, max_ntree_per_split = get_infos_from_gbms( gbms, min_len_features=x_train.shape[1])
    n_models = len(used_features)

    # Train embedding model
    print("\nTrain embedding model...")
    emb_model = EmbeddingModel(n_models, max_ntree_per_split, n_output=y_train)
   
    optimizer = AdamW(emb_model.parameters(), lr=config.config['emb_lr'], weight_decay=config.config['l2_reg'])
    tree_outputs = np.asarray(tree_outputs).reshape((n_models, leaf_preds.shape[0])).transpose((1, 0))

    trainModel(emb_model, leaf_preds, y_train, tree_outputs, test_leaf_preds, y_test,
               optimizer, epochs=config.config['emb_epochs'])
    output_w, output_b, tree_outputs = outputFromEmbeddingModel(emb_model, leaf_preds, y_train, n_models)

    # Train DeepGBM model
    print("\nTrain DeepGBM model...")
    deepgbm_model = DeepGBM(nume_input_size=x_train.shape[1],
                            used_features=np.asarray(used_features, dtype=np.int64),
                            output_w=output_w,
                            output_b=output_b,
                            cate_field_size=train_cat.shape[1],
                            feature_sizes=feature_sizes)
   

    optimizer = AdamW(deepgbm_model.parameters(), lr=config.config['lr'], weight_decay=config.config['l2_reg'],
                      amsgrad=False, model_decay_opt=deepgbm_model, weight_decay_opt=config.config['l2_reg_opt'],
                      key_opt='deepgbm')

    x_train = np.concatenate([x_train, np.zeros((x_train.shape[0], 1), dtype=np.float32)], axis=-1)
    x_test = np.concatenate([x_test, np.zeros((x_test.shape[0], 1), dtype=np.float32)], axis=-1)

    _, _, loss_history, val_loss_history = trainModel(deepgbm_model, x_train, y_train, tree_outputs, x_test, y_test,
                                                      optimizer,
                                                      train_x_cat=train_cat, test_x_cat=test_cat,
                                                      epochs=config.config['epochs'],
                                                      early_stopping_rounds=config.config['early_stopping'],
                                                      save_model=save_model)

   # model, optimizer, loss_history, val_loss_history

    return deepgbm_model, optimizer, loss_history, val_loss_history


'''
    Evaluate the model (metric depends on given task)
    
    Parameters:
        - model: trained model
        - test_num: data and label preprocessed for testing
        - test_cat: data preprocessed for testing CatNN
        
    Returns:
        - test_loss: loss computed on the testing data
        - preds: predicitions on test data
'''
#print('Left only evaluation of the system of the given mothed')

def evaluate(model, test_num, test_cat):
    device = config.config['device']

    x_test, y_test = test_num
    x_test = np.concatenate([x_test, np.zeros((x_test.shape[0], 1), dtype=np.float32)], axis=-1)
    print('\n################### Evaluate model #######################################\n')
   
    test_loss, preds = evaluateModel(model, x_test, y_test, test_cat)


        # define min and max values of the original dataset
    max_value = 100
    min_value = 5

    # define a normalized value between 0 and 1
    # x_normalized = 0.75

    # denormalize the value back to its original scale
    x = (preds* (max_value - min_value)) + min_value

    # print(np.array(x), ' in Evalue method get these values')# output: 80.0


    metric = eval_metrics(config.config['task'], y_test, preds)
    printMetric(config.config['task'], metric, test_loss) 


'''
    Make predicitions on new data
    Parameters:
        - model: trained model
        - data: new unlabeled test data
        - ce: CatEncoder from the preprocessing
        - ne: NumEncoder from the preprocessing
    Returns:
        - predicitions on data
'''


def predict(model, data, ce, ne):
    # Transform data properly
    data_cat = ce.transform(data.copy()).astype('int32')
    data_num = ne.transform(data.copy())

    data_num = np.concatenate([data_num, np.zeros((data_num.shape[0], 1), dtype=np.float32)], axis=-1)
    # print(data_num)
    print('\n################### Make predictions #######################################\n')

    return makePredictions(model, data_num, data_cat)

      # makePredictions(model, test_x, test_cat)
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
   
####################################################################################################################################
####################################################################################################################################
    data = data.drop('CUSTOMER_ID', axis=1)
    
    data = data.loc[1:1705200, :]
    row_sums = data.sum(axis=1)
    # assign the sum of rows to column
    df1 = data.assign(row_sum=row_sums)
    

    # max_value = df1['row_sum'].max()
    # min_value = df1['row_sum'].min()
    # print(max_value, min_value, 'your interval of these given one')


   #Classification based on the given interval
    bins = [-40.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 
            500001.0, 600001.0, 700001.0, 1000000.0, 1000000001.0, 1297879501.0, 5000000001.0]

    labels = [5 , 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,  80, 85, 90, 95, 100]
    df1['row_sum'] = pd.cut(df1['row_sum'], bins=bins, labels=labels)
    
     # data = data.drop('Customer_ID', axis=1)
    df1 = df1.drop('target', axis=1)

    # data = df1.rename(columns={'row_sum': 'target'})
    df1['row_sum'] = df1['row_sum'].astype('category')
    df1['row_sum'] = df1['row_sum'].cat.codes
    # create a sample dataset
    max_value = df1['row_sum'].max()
    min_value = df1['row_sum'].min()
    
    df1['target'] = (df1['row_sum'].values - min_value) / (max_value - min_value)
    
    # print(df1.head(60))
    
    df1 = df1.drop('row_sum', axis=1)
    data = df1
    
#######################################################################################################################################
#######################################################################################################################################

    # data = data.drop('Customer_ID', axis=1)
    dt = data.loc[2:11, :]
    # print(dt)
    train_num, test_num, train_cat, test_cat, feature_sizes, ce, ne = preprocess(data, num_col, cat_col, label_col)

    #(train_x, train_y), (test_x, test_y), train_x_cat, test_x_cat, feature_sizes, ce, ne
    deepgbm_model, optimizer, loss_history, val_loss_history = train(train_num, test_num, train_cat, test_cat, feature_sizes)
    torch.save(deepgbm_model.state_dict(), "model12.pth")
    deepgbm_model.load_state_dict(torch.load("model12.pth"))
     
    torch.save(deepgbm_model, "model12.pt")
    
    the_model = torch.load("model12.pt")

    evaluate(deepgbm_model, test_num, test_cat)
    predict(the_model, dt, ce, ne)

if __name__ == '__main__':
    main()