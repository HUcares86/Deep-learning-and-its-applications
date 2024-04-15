from torch.optim import AdamW
import warnings
import random
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# import add train data ---------------------------------------------------------------------------------------------
crypto_df_add = pd.read_csv('/home/shelley/Desktop/hucares/DL_final2/add_train.csv')
print("crypto_df_add before")
print(crypto_df_add)
print("Total Null Target Rows = " ,crypto_df_add["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(crypto_df_add["Target"].isnull().sum()*100 / crypto_df_add.shape[0] ))
print("--------"*15)


crypto_df_add['date'] = pd.to_datetime(crypto_df_add['timestamp'], unit='s')    # timestamp
df0_add = crypto_df_add.drop(crypto_df_add[(crypto_df_add['Asset_ID'] != 1) & (crypto_df_add['Asset_ID'] != 6)].index) # delete other assets
df0_add = df0_add[df0_add['date'] > '2021-09-21 00:00:00']  # keep data after 2021-09-21 00:00:00
print("df0")
print(df0_add)
print("Total Null Target Rows = " ,df0_add["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df0_add["Target"].isnull().sum()*100 / df0_add.shape[0] ))
print("--------"*15)

# import train data ---------------------------------------------------------------------------------------------
crypto_df1 = pd.read_csv('/home/shelley/Desktop/hucares/DL_final2/train.csv')
print("crypto_df1")
print(crypto_df1)
print("Total Null Target Rows = " ,crypto_df1["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(crypto_df1["Target"].isnull().sum()*100 / crypto_df1.shape[0] ))
print("--------"*15)

crypto_df1['date'] = pd.to_datetime(crypto_df1['timestamp'], unit='s')# timestamp

df1 = crypto_df1.drop(crypto_df1[(crypto_df1['Asset_ID'] != 1) & (crypto_df1['Asset_ID'] != 6)].index)# delete other assets
print("df1 1 ")
print(df1)
print("Total Null Target Rows = " ,df1["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df1["Target"].isnull().sum()*100 / df1.shape[0] ))
print("--------"*15)




# create split data

# 2021-09-21 -----------------------------------------------------------------------------------------
# train: 2018_01_01 ~ 2021_03_21
# valid: 2021_03_21 ~ 22022-01-24

# test: 2022-05-24 ~ 2022-12-04

df_a = pd.concat([df1,df0_add],axis=0)
df_a = df_a[df_a['date'] < '2022-05-24 00:00:00']

df_a_1 = df_a[df_a['date'] < '2022-01-24 00:00:00']
df_a_1 = df_a_1.drop(df_a_1[(df_a_1['Asset_ID'] != 1) & (df_a_1['Asset_ID'] != 6)].index)
print("df_a_1 ")
# print(df_a_1)
print("Total Null Target Rows = " ,df_a_1["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_a_1["Target"].isnull().sum()*100 / df_a_1.shape[0] ))
print("--------"*15)
df_a_1 = df_a_1.merge(pd.read_csv('/home/shelley/Desktop/hucares/DL_final2/asset_details.csv'), on="Asset_ID", how="left")
# print(df_a)
# print("--------"*15)
df_a_1.to_csv(os.path.join('/home/shelley/Desktop/hucares/DL_final2/','train_before_2022_01_24_data.csv')) #save to file


df_a_2 = df_a[df_a['date'] > '2022-01-04 00:00:00']
df_a_2 = df_a_2.drop(df_a_2[(df_a_2['Asset_ID'] != 1) & (df_a_2['Asset_ID'] != 6)].index)
print("df_a_2 ")
# print(df_a_2)
print("Total Null Target Rows = " ,df_a_2["Target"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_a_2["Target"].isnull().sum()*100 / df_a_2.shape[0] ))
print("--------"*15)
df_a_2 = df_a_2.merge(pd.read_csv('/home/shelley/Desktop/hucares/DL_final2/asset_details.csv'), on="Asset_ID", how="left")
# print(df_a)
# print("--------"*15)
df_a_2.to_csv(os.path.join('/home/shelley/Desktop/hucares/DL_final2/','test_2022_01_04_to_2022_05_04_data.csv')) #save to file

