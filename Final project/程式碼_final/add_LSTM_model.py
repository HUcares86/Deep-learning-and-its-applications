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
# import gresearch_crypto
import time
import datetime
import math
import pickle
import gc
from scipy.stats.stats import pearsonr   
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import torch
sns.set()
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


# train data input ---------------------------------------------------
TRAIN_CSV = "/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/feat.csv"
# training_pred_csv = "/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/all_training_predict.csv"
training_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/train_feat_y_true.csv'
SUPPLE_TRAIN_CSV = '/home/shelley/Desktop/hucares/DL_final2/supplemental_train.csv'

# ASSET_DETAILS_CSV = '/home/shelley/Desktop/hucares/DL_final2/asset_details.csv'
ASSET_DETAILS_CSV = '/home/shelley/Desktop/hucares/DL_final2/asset_details_no_weight.csv'
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

df_train = pd.read_csv(TRAIN_CSV, index_col=0)
# print('--'*50)
# print(df_train)
# print('')

df_train_y_true = pd.read_csv(training_y_true_csv, index_col=0)
# print('--'*50)
# print(df_train_y_true)
# print('')


# testing data input -----------------------------------------------------------------
Test_csv = "/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/test_feat.csv"
Test_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_add_feature/test_feat_y_true.csv'
df_test= pd.read_csv(Test_csv, index_col=0)
# print('--'*50)
# print(df_test)
# print('')

df_test_y_true = pd.read_csv(Test_y_true_csv, index_col=0)
# print('--'*50)
# print(df_test_y_true)
# print('')

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
# df_test_total = df_test_total.dropna()
print("Total Null Target Rows = " ,df_test_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_test_total["Target_1"].isnull().sum()*100 / df_test_total.shape[0] ))
print("--------"*15)

df_train_total = df_train
df_train_total['Target_1'] = df_train_y_true['Target_1']
df_train_total['Target_6'] = df_train_y_true['Target_6']
# df_train_total = df_train_total.dropna()
print("Total Null Target Rows = " ,df_train_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_train_total["Target_1"].isnull().sum()*100 /df_train_total.shape[0] ))
print("--------"*15)

df_test_y_true = pd.DataFrame()
df_test_y_true['Target_1'] = df_test_total['Target_1'].values
df_test_y_true['Target_6'] = df_test_total['Target_6'].values
df_test = df_test_total.drop('Target_1',axis =1)
df_test = df_test_total.drop('Target_6',axis =1)

df_train_y_true = pd.DataFrame()
df_train_y_true['Target_1'] = df_train_total['Target_1'].values
df_train_y_true['Target_6'] = df_train_total['Target_6'].values
df_train = df_train_total.drop('Target_1',axis =1)
df_train = df_train_total.drop('Target_6',axis =1)


print(df_test_y_true)

print(df_test)

print(df_train_y_true)

print(df_train)
feat = df_train.values
test_feat = df_test.values

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
# LSTM --------------------------------------------------------------------------------------------------------------
#split into train valid
split =round(df_train_total.shape[0]*0.9)
train_X , train_y = feat[:split, :] , feat[:split, :]
test_X , test_y = feat[split:, :] , feat[split:, :]

n_features = feat.shape[2]
print ("train_X.shape" , train_X.shape)
print ("train_y.shape" , train_y.shape)
print ("test_X.shape" , test_X.shape)
print ("test_y.shape" , test_y.shape)
print ("n_features" , n_features)
    
#optimizer learning rate
opt = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mse' , optimizer=opt , metrics=['mse'])

# Fit network
history = model.fit(train_X , train_y , epochs=10 , steps_per_epoch=50 , verbose=1 ,validation_data=(test_X, test_y) ,shuffle=False)
# ========================================================================================================================================================


y_pred_inv = prep_data(df_test_total )
y_pred_inv.to_csv(os.path.join('/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/y_pred_inv.csv''),index=False) #save to file
print('--'*100)
print('')

rms_bit = mean_squared_error(df_test_y_true , y_pred_inv[0,:], squared=False)
pearsonr_bit_score = pearsonr(df_test_y_true ,y_pred_inv[0,:])
corr_bit_score = np.corrcoef(df_test_y_true ,y_pred_inv[0,:])
print(f'pearsonr score of bit ID={j}_fold{i} is {corr_bit_score}')
print(f'corr score of bit ID={j}_fold{i} is {corr_bit_score}')
print(f'bit (ID=1) RMSE is {rms_bit}')
    
rms_eth = mean_squared_error(df_test_y_true , y_pred_inv[1,:], squared=False)
pearsonr_eth_score = pearsonr(df_test_y_true ,y_pred_inv[1,:])
corr_eth_score = np.corrcoef(df_test_y_true ,y_pred_inv[1,:])
print(f'pearsonr score of eth ID={j}_fold{i} is {corr_bit_score}')
print(f'corr score of eth ID={j}_fold{i} is {corr_bit_score}')
print(f'bit (ID=1) RMSE is {rms_bit}')
