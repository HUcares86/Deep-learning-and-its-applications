import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorflow import keras
import torch.nn.functional as F
import tensorflow as tf
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from scipy.stats.stats import pearsonr   
from tqdm import tqdm
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
# import gresearch_crypto
import time
import datetime
import math
import pickle
import gc
from scipy.stats.stats import pearsonr   
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

n_fold = 7
seed0 = 8586
use_supple_for_train =True
# If True, the period used to evaluate Public LB will not be used for training.
# Set to False on final submission.
not_use_overlap_to_train = False
lags = [60,300,900]
lags_interval = [[5,10],[11,20],[21,40],[41,60],[61,80],[81,100],[101,200]]
id_list = [1,6]

params = {
    'early_stopping_rounds': 50,
    'objective': 'regression',
    'metric': 'rmse',
#     'metric': 'None',
    'boosting_type': 'gbdt',
    'max_depth': 5,
    'verbose': -1,
    'max_bin':600,
    'min_data_in_leaf':50,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'subsample_freq': 1,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_fraction_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'extra_trees': True,
    'extra_seed': seed0,
    'zero_as_missing': True,
    "first_metric_only": True
         }
# train data input ---------------------------------------------------
TRAIN_CSV = "/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/feat.csv"
training_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/train_feat_y_true.csv'
SUPPLE_TRAIN_CSV = '/home/shelley/Desktop/hucares/DL_final2/supplemental_train.csv'
ASSET_DETAILS_CSV = '/home/shelley/Desktop/hucares/DL_final2/asset_details_no_weight.csv'
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

df_train = pd.read_csv(TRAIN_CSV, index_col=0)

df_train_y_true = pd.read_csv(training_y_true_csv, index_col=0)


# testing data input -----------------------------------------------------------------
Test_csv = "/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/test_feat.csv"
Test_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/test_feat_y_true.csv'
df_test= pd.read_csv(Test_csv, index_col=0)

df_test_y_true = pd.read_csv(Test_y_true_csv, index_col=0)

# add train test pred -----------------------------------------------------------------
for i in range(7):
    for j in id_list:
        if j == 1:
            # /home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature
            train_bit_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/train_pred_id_{j}_fold{i}.csv'
            df_bit_train = pd.read_csv(train_bit_csv, index_col=0).astype('float16')
            df_train[f'train_pred_id_{j}_fold{i}'] = df_bit_train
            
            test_bit_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/test_pred_id_{j}_fold{i}.csv'
            df_bit_test = pd.read_csv(test_bit_csv, index_col=0).astype('float16')
            df_test[f'test_pred_id_{j}_fold{i}'] = df_bit_test


        if j == 6:
            train_eth_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/train_pred_id_{j}_fold{i}.csv'
            df_eth_train = pd.read_csv(train_eth_csv, index_col=0).astype('float16')
            df_train[f'train_pred_id_{j}_fold{i}'] = df_eth_train
            
            test_eth_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/test_pred_id_{j}_fold{i}.csv'
            df_eth_test = pd.read_csv(test_eth_csv, index_col=0).astype('float16')
            df_test[f'test_pred_id_{j}_fold{i}'] = df_eth_test



df_test_total = df_test
df_test_total['Target_1'] = df_test_y_true['Target_1']
df_test_total['Target_6'] = df_test_y_true['Target_6']

print("Total Null Target Rows = " ,df_test_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_test_total["Target_1"].isnull().sum()*100 / df_test_total.shape[0] ))
print("--------"*15)

df_train_total = df_train
df_train_total['Target_1'] = df_train_y_true['Target_1']
df_train_total['Target_6'] = df_train_y_true['Target_6']

print("Total Null Target Rows = " ,df_train_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_train_total["Target_1"].isnull().sum()*100 /df_train_total.shape[0] ))
print("--------"*15)

df_test_y_true = pd.DataFrame()
df_test_y_true['Target_1'] = df_test_total['Target_1'] 
df_test_y_true['Target_6'] = df_test_total['Target_6'] 

df_train_y_true = pd.DataFrame()
df_train_y_true['Target_1'] = df_train_total['Target_1'] 
df_train_y_true['Target_6'] = df_train_total['Target_6'] 



print(df_test_y_true)

print(df_test)

print(df_train_y_true)

print(df_train)
feat = df_train
test_feat = df_test

not_use_features_train = ['timestamp', 'train_flg']
not_use_features_train.append(f'Target_{1}')
not_use_features_train.append(f'Target_{6}')
features = feat.columns 
features = features.drop(not_use_features_train)
features = list(features)

not_use_features_test = [f'Target_{1}', f'Target_{6}']
features_test = test_feat.columns 
features_test = features_test.drop(not_use_features_test)
features_test = list(features_test)
test_feat = df_test.loc[:,features_test]
print(test_feat)
print("--------------")
# define the evaluation metric  --------------------------------------------------------------------------------------------------------------
def correlation(a, train_data):
    
    b = train_data.get_label()
    
    a = np.ravel(a)
    b = np.ravel(b)

    len_data = len(a)
    mean_a = np.sum(a) / len_data
    mean_b = np.sum(b) / len_data
    var_a = np.sum(np.square(a - mean_a)) / len_data
    var_b = np.sum(np.square(b - mean_b)) / len_data

    cov = np.sum((a * b))/len_data - mean_a*mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return 'corr', corr, True

# For CV score calculation
def corr_score(pred, valid):
    len_data = len(pred)
    mean_pred = np.sum(pred) / len_data
    mean_valid = np.sum(valid) / len_data
    var_pred = np.sum(np.square(pred - mean_pred)) / len_data
    var_valid = np.sum(np.square(valid - mean_valid)) / len_data

    cov = np.sum((pred * valid))/len_data - mean_pred*mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr

# For CV score calculation
def wcorr_score(pred, valid, weight):
    len_data = len(pred)
    sum_w = np.sum(weight)
    mean_pred = np.sum(pred * weight) / sum_w
    mean_valid = np.sum(valid * weight) / sum_w
    var_pred = np.sum(weight * np.square(pred - mean_pred)) / sum_w
    var_valid = np.sum(weight * np.square(valid - mean_valid)) / sum_w

    cov = np.sum((pred * valid * weight)) / sum_w - mean_pred*mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr

def get_time_series_cross_val_splits(data, cv = n_fold, embargo = 3750):
    all_train_timestamps = data['timestamp'].unique()
    len_split = len(all_train_timestamps) // cv
    test_splits = [all_train_timestamps[i * len_split:(i + 1) * len_split] for i in range(cv)]
    # fix the last test split to have all the last timestamps, in case the number of timestamps wasn't divisible by cv
    rem = len(all_train_timestamps) - len_split*cv
    if rem>0:
        test_splits[-1] = np.append(test_splits[-1], all_train_timestamps[-rem:])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        # get all of the timestamps that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_timestamps if not (test_split_min <= int(e) <= test_split_max)]
        # embargo the train split so we have no leakage. Note timestamps are expressed in seconds, so multiply by 60
        embargo_sec = 60*embargo
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo_sec and abs(int(e) - test_split_min) > embargo_sec]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


# training valid testing --------------------------------------------------------------------------------------------------------------
def get_Xy_and_model_for_asset(df_proc, asset_id):
    df_proc = df_proc.loc[  (df_proc[f'Target_{asset_id}'] == df_proc[f'Target_{asset_id}'])  ]
    if not_use_overlap_to_train:
        df_proc = df_proc.loc[  (df_proc['train_flg'] == 1)  ]
    print(df_proc.columns )
    train_test_zip = get_time_series_cross_val_splits(df_proc, cv = n_fold, embargo = 3750)
    print("entering time series cross validation loop")
    importances = []
    oof_pred = []
    oof_valid = []
    
    all_training_predict = pd.DataFrame()
    
    for split, train_test_split in enumerate(train_test_zip):
        gc.collect()
        
        print(f"doing split {split+1} out of {n_fold}")
        train_split, test_split = train_test_split
        train_split_index = df_proc['timestamp'].isin(train_split)
        test_split_index = df_proc['timestamp'].isin(test_split)
    
        train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
                                    df_proc.loc[train_split_index, f'Target_{asset_id}'].values, 
                                    feature_name = features, 
                                   )
        val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features], 
                                  df_proc.loc[test_split_index, f'Target_{asset_id}'].values, 
                                  feature_name = features, 
                                 )

        print(f"number of train data: {len(df_proc.loc[train_split_index])}")
        print(f"number of val data:   {len(df_proc.loc[test_split_index])}")

        model = lgb.train(params = params,
                          train_set = train_dataset, 
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['tr', 'vl'],
                          num_boost_round = 5000,
                          verbose_eval = 100,     
                          feval = correlation,
                         )
        importances.append(model.feature_importance(importance_type='gain'))
        
        file = f'trained_model_id{asset_id}_fold{split}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print(f"Trained model was saved to 'trained_model_id{asset_id}_fold{split}.pkl'")
        print("")
        
        # print(df_proc.loc[test_split_index, features])
        oof_pred += list(  model.predict(df_proc.loc[test_split_index, features])        )
        oof_valid += list(   df_proc.loc[test_split_index, f'Target_{asset_id}'].values    )
        
        # all_training_predict[f'test_pred_id_{asset_id}_fold{split}'] = train_pred
        # print(len(train_pred))
        
        # get testing predict
        test_pred = list( model.predict(test_feat))
        for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
            # bitcoin
            if asset_id == 1:
                df_test_pred = pd.DataFrame(test_pred)
                df_test_pred.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/add_model',f'final_test_pred_id_{asset_id}_fold{split}.csv')) #save to file   
               
            # eth
            if asset_id == 6:
                df_test_pred = pd.DataFrame(test_pred)
                df_test_pred.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/add_model',f'final_test_pred_id_{asset_id}_fold{split}.csv')) 

    return oof_pred, oof_valid ,test_pred



oof = [ [] for id in range(14)]
all_oof_pred = []
all_oof_valid = []
all_oof_weight = []

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    
    oof_pred, oof_valid, test_pred  = get_Xy_and_model_for_asset(feat, asset_id)
    
    weight_temp = float( df_asset_details.loc[  df_asset_details['Asset_ID'] == asset_id  , 'Weight'   ]  )

    all_oof_pred += oof_pred
    all_oof_valid += oof_valid
    all_oof_weight += [weight_temp] * len(oof_pred)
    
    # calculate MSE & corr
    oof[asset_id] = corr_score(     np.array(oof_pred)   ,    np.array(oof_valid)    )
    MSE1 = np.square(np.subtract(oof_valid,oof_pred)).mean() 
    RMSE1 = math.sqrt(MSE1)
    
    print(f'OOF corr score of {asset_name} (ID={asset_id}) is {oof[asset_id]:.5f}. (Weight: {float(weight_temp):.5f})')
    print(f'of {asset_name} (ID={asset_id}) MSE is {MSE1}. RMSE is {RMSE1}')
    print('')
    print('')
    
    
    
# ========================================================================================================================================================


