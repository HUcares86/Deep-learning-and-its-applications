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

from tqdm import tqdm

n_fold = 5
seed0 = 8586
use_supple_for_train =True
not_use_overlap_to_train = False

# data input --------------------------------------------------------------------------------------------------------------

TRAIN_CSV = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/train_before_2022_05_24_data.csv'
SUPPLE_TRAIN_CSV = '/home/shelley/Desktop/hucares/DL_final2/supplemental_train.csv'
ASSET_DETAILS_CSV = '/home/shelley/Desktop/hucares/DL_final2/asset_details_no_weight.csv'
Test_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/test_2022_05_24_to_now_data.csv'



# params  --------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 350)

lags = [60,300,900]


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

# reduce memory   ---------------------------------------------------------------------------------------------
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# asset_details ---------------------------------------------------------------------------------------------
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")


# training data --------------------------------------------------------------------------------------------------------------
df_train = pd.read_csv(TRAIN_CSV, usecols=['timestamp','Asset_ID', 'Close', 'Target'])
if use_supple_for_train:    
    df_supple = pd.read_csv(SUPPLE_TRAIN_CSV, usecols=['timestamp','Asset_ID', 'Close', 'Target'])
#     display(df_supple)
    df_train = pd.concat([df_train, df_supple])
    del df_supple
df_train = reduce_mem_usage(df_train)

train_merged = pd.DataFrame()
train_merged[df_train.columns] = 0

train_merged = train_merged.merge(df_train.loc[df_train["Asset_ID"] == 1, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(1)])
train_merged = train_merged.merge(df_train.loc[df_train["Asset_ID"] == 6, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(6)])       
        
train_merged = train_merged.drop(df_train.columns.drop("timestamp"), axis=1)
train_merged = train_merged.sort_values('timestamp', ascending=True)


# Set an upper limit on the number of fills, since there may be long term gaps.
#     print(id, train_merged[f'Close_{id}'].isnull().sum())   # Number of missing before forward fill
train_merged[f'Close_{1}'] = train_merged[f'Close_{1}'].fillna(method='ffill', limit=100)
train_merged[f'Close_{6}'] = train_merged[f'Close_{6}'].fillna(method='ffill', limit=100)


# add features to training data
def get_features(df, train=True):   
    if train == True:
        totimestamp = lambda s: np.int32(time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()))
        valid_window = [totimestamp("21/03/2021")]
        df['train_flg'] = np.where(df['timestamp']>=valid_window[0], 0,1)

    for lag in lags:
        df[f'log_close/mean_{lag}_id{1}'] = np.log( np.array(df[f'Close_{1}']) /  np.roll(np.append(np.convolve( np.array(df[f'Close_{1}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)  )
        df[f'log_return_{lag}_id{1}']     = np.log( np.array(df[f'Close_{1}']) /  np.roll(np.array(df[f'Close_{1}']), lag)  )
        df[f'log_close/mean_{lag}_id{6}'] = np.log( np.array(df[f'Close_{6}']) /  np.roll(np.append(np.convolve( np.array(df[f'Close_{6}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)  )
        df[f'log_return_{lag}_id{6}']     = np.log( np.array(df[f'Close_{6}']) /  np.roll(np.array(df[f'Close_{6}']), lag)  )
        
    for lag in lags:
        df[f'mean_close/mean_{lag}'] =  np.mean(df.iloc[:,df.columns.str.startswith(f'log_close/mean_{lag}_id')], axis=1)
        df[f'mean_log_returns_{lag}'] = np.mean(df.iloc[:,df.columns.str.startswith(f'log_return_{lag}_id')] ,    axis=1)
        
        df[f'log_close/mean_{lag}-mean_close/mean_{lag}_id{1}'] = np.array( df[f'log_close/mean_{lag}_id{1}']) - np.array( df[f'mean_close/mean_{lag}']  )
        df[f'log_return_{lag}-mean_log_returns_{lag}_id{1}']    = np.array( df[f'log_return_{lag}_id{1}'])     - np.array( df[f'mean_log_returns_{lag}'] )
        df[f'log_close/mean_{lag}-mean_close/mean_{lag}_id{6}'] = np.array( df[f'log_close/mean_{lag}_id{6}']) - np.array( df[f'mean_close/mean_{lag}']  )
        df[f'log_return_{lag}-mean_log_returns_{lag}_id{6}']    = np.array( df[f'log_return_{lag}_id{6}'])     - np.array( df[f'mean_log_returns_{lag}'] )
        
    if train == True:
        oldest_use_window = [totimestamp("12/01/2019")]
        df = df[  df['timestamp'] >= oldest_use_window[0]   ]

    return df


# get added features training data, del things we don't need
feat = get_features(train_merged)
not_use_features_train = ['timestamp', 'train_flg']
not_use_features_train.append(f'Target_{1}')
not_use_features_train.append(f'Target_{6}')
features = feat.columns 
features = features.drop(not_use_features_train)
features = list(features)
del train_merged
del df_train
print('train_feat.columns: ',len(features))


# testing data -----------------------------------------------------------------------------------------------------------------------------------------
df_test = pd.read_csv(Test_csv, usecols=['timestamp','Asset_ID', 'Close', 'Target'])
# df_test = reduce_mem_usage(df_test)

test_merged = pd.DataFrame()
test_merged[df_test.columns] = 0
test_merged = test_merged.merge(df_test.loc[df_test["Asset_ID"] == 1, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(1)])
test_merged = test_merged.merge(df_test.loc[df_test["Asset_ID"] == 6, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(6)])             
test_merged = test_merged.drop(df_test.columns.drop("timestamp"), axis=1)
test_merged = test_merged.sort_values('timestamp', ascending=True)

test_merged[f'Close_{1}'] = test_merged[f'Close_{1}'].fillna(method='ffill', limit=100)
test_merged[f'Close_{6}'] = test_merged[f'Close_{6}'].fillna(method='ffill', limit=100)



# add features to testing data
def get_test_features(df):      
    for lag in lags:
        df[f'log_close/mean_{lag}_id{1}'] = np.log( np.array(df[f'Close_{1}']) /  np.roll(np.append(np.convolve( np.array(df[f'Close_{1}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)  )
        df[f'log_return_{lag}_id{1}']     = np.log( np.array(df[f'Close_{1}']) /  np.roll(np.array(df[f'Close_{1}']), lag)  )
        df[f'log_close/mean_{lag}_id{6}'] = np.log( np.array(df[f'Close_{6}']) /  np.roll(np.append(np.convolve( np.array(df[f'Close_{6}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)  )
        df[f'log_return_{lag}_id{6}']     = np.log( np.array(df[f'Close_{6}']) /  np.roll(np.array(df[f'Close_{6}']), lag)  )
        
    for lag in lags:
        df[f'mean_close/mean_{lag}'] =  np.mean(df.iloc[:,df.columns.str.startswith(f'log_close/mean_{lag}_id')], axis=1)
        df[f'mean_log_returns_{lag}'] = np.mean(df.iloc[:,df.columns.str.startswith(f'log_return_{lag}_id')] ,    axis=1)
        
        df[f'log_close/mean_{lag}-mean_close/mean_{lag}_id{1}'] = np.array( df[f'log_close/mean_{lag}_id{1}']) - np.array( df[f'mean_close/mean_{lag}']  )
        df[f'log_return_{lag}-mean_log_returns_{lag}_id{1}']    = np.array( df[f'log_return_{lag}_id{1}'])     - np.array( df[f'mean_log_returns_{lag}'] )
        df[f'log_close/mean_{lag}-mean_close/mean_{lag}_id{6}'] = np.array( df[f'log_close/mean_{lag}_id{6}']) - np.array( df[f'mean_close/mean_{lag}']  )
        df[f'log_return_{lag}-mean_log_returns_{lag}_id{6}']    = np.array( df[f'log_return_{lag}_id{6}'])     - np.array( df[f'mean_log_returns_{lag}'] )
    
    return df

# testing features
test_feat = get_test_features(test_merged)
test_bit_y_true = list(test_feat[f'Target_{1}'])
test_eth_y_true = list(test_feat[f'Target_{6}'])
test_feat = test_feat.drop('Target_1',axis =1)
test_feat = test_feat.drop('timestamp',axis =1)
test_feat = test_feat.drop('Target_6',axis =1)
print('test_feat.columns: ',len(test_feat.columns))



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

# from: https://blog.amedama.jp/entry/lightgbm-cv-feature-importance
# (used in nyanp's Optiver solution)
def plot_importance(importances, features_names = features, PLOT_TOP_N = 20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()


def get_time_series_cross_val_splits(data, cv = n_fold, embargo = 3750):
    all_train_timestamps = data['timestamp'].unique()
    len_split = len(all_train_timestamps) // cv
    test_splits = [all_train_timestamps[i * len_split:(i + 1) * len_split] for i in range(cv)]
    rem = len(all_train_timestamps) - len_split*cv
    if rem>0:
        test_splits[-1] = np.append(test_splits[-1], all_train_timestamps[-rem:])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        train_split_not_embargoed = [e for e in all_train_timestamps if not (test_split_min <= int(e) <= test_split_max)]
        embargo_sec = 60*embargo
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo_sec and abs(int(e) - test_split_min) > embargo_sec]
        train_splits.append(train_split)
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


# training valid testing --------------------------------------------------------------------------------------------------------------
def get_Xy_and_model_for_asset(df_proc, asset_id):
    df_proc = df_proc.loc[  (df_proc[f'Target_{asset_id}'] == df_proc[f'Target_{asset_id}'])  ]
    if not_use_overlap_to_train:
        df_proc = df_proc.loc[  (df_proc['train_flg'] == 1)  ]

    train_test_zip = get_time_series_cross_val_splits(df_proc, cv = n_fold, embargo = 3750)
    print("entering time series cross validation loop")
    importances = []
    oof_pred = []
    oof_valid = []
    
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
            
        oof_pred += list(  model.predict(df_proc.loc[test_split_index, features])        )
        oof_valid += list(   df_proc.loc[test_split_index, f'Target_{asset_id}'].values    )
        
        # get testing predict
        test_pred = list( model.predict(test_feat))
        for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
            # bitcoin
            if asset_id == 1:
                df_test_pred = pd.DataFrame(test_pred)
                df_test_pred.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/old/',f'test_pred_id_{asset_id}_fold{split}.csv')) #save to file   

        
            # eth
            if asset_id == 6:
                df_test_pred = pd.DataFrame(test_pred)
                df_test_pred.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/old/',f'test_pred_id_{asset_id}_fold{split}.csv')) 
 

    # plot feature importance
    plot_importance(np.array(importances),features, PLOT_TOP_N = 50, figsize=(50, 20))#, figsize=(10, 5
    plt.savefig(f'ID={asset_id}_importance.png')  
    plt.clf()
    

    return oof_pred, oof_valid


oof = [ [] for id in range(14)]
all_oof_pred = []
all_oof_valid = []
all_oof_weight = []

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    
    oof_pred, oof_valid  = get_Xy_and_model_for_asset(feat, asset_id)
    
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

