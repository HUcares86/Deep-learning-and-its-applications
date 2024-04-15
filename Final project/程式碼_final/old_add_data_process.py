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
id_list = [1,6]

# get y true  --------------------------------------------------------------------------------------------------------------
Test_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/test_2022_05_24_to_now_data.csv'
df_test = pd.read_csv(Test_csv, usecols=['timestamp','Asset_ID', 'Close', 'Target'])
test_merged = pd.DataFrame()
test_merged[df_test.columns] = 0
test_merged = test_merged.merge(df_test.loc[df_test["Asset_ID"] == 1, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(1)])
test_merged = test_merged.merge(df_test.loc[df_test["Asset_ID"] == 6, ['timestamp', 'Close','Target']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(6)])             
test_merged = test_merged.drop(df_test.columns.drop("timestamp"), axis=1)
test_merged = test_merged.sort_values('timestamp', ascending=True)


test_merged[f'Close_{1}'] = test_merged[f'Close_{1}'].fillna(method='ffill', limit=100)
test_merged[f'Close_{6}'] = test_merged[f'Close_{6}'].fillna(method='ffill', limit=100)

total_bit_true = []
total_bit_test = []
avg_bit_test = []

total_eth_true = []
total_eth_test = []
avg_eth_test = []

all_test_predict = pd.DataFrame()

for i in range(5):
    for j in id_list:

        if j == 1:
            #/home/shelley/Desktop/hucares/DL_final2/2022_05_24_to_now/2022_05_24_to_now_old
            Test_bit_csv = f'/home/shelley/Desktop/hucares/DL_final2/test_pred_id_{j}_fold{i}.csv'
            df_bit_test = pd.read_csv(Test_bit_csv, index_col=0).astype('float16')
            df_bit_test["test_bit_y_true"] = test_merged[f'Target_{1}']
            df_bit_test.rename(columns = {"0":'test_bit_pred'}, inplace = True)
            df_bit_test=df_bit_test.dropna()
            # print("df_bit_test : \n")
            # print(df_bit_test)

            
            test_bit_y_true = list(df_bit_test["test_bit_y_true"])
            test_bit_pred = list(df_bit_test['test_bit_pred'])
            total_bit_true += test_bit_y_true 
            total_bit_test += test_bit_pred
            if i == 0:
                avg_bit_test = test_bit_pred
                all_test_predict[f'test_bit_y_true'] = test_bit_y_true
                
            else:
                avg_bit_test = np.sum([avg_bit_test,test_bit_pred], axis=0)
            all_test_predict[f'test_pred_id_{j}_fold{i}'] = test_bit_pred
            
            rms_bit = mean_squared_error(test_bit_y_true , test_bit_pred, squared=False)
            pearsonr_bit_score = pearsonr(test_bit_y_true ,test_bit_pred)
            corr_bit_score = np.corrcoef(test_bit_y_true ,test_bit_pred)
            print(f'pearsonr score of bit ID={j}_fold{i} is {corr_bit_score}')
            print(f'corr score of bit ID={j}_fold{i} is {corr_bit_score}')
            print(f'bit (ID=1) RMSE is {rms_bit}')
    
            
            # plot test predicts
            plt.plot(test_bit_y_true,'b')
            plt.savefig(f'ID_{j}_fold{i}_test_bit_y_true.png')  
            plt.clf()
            plt.plot(test_bit_pred,'b')
            plt.savefig(f'ID={j}_fold{i}_test_bit_pred.png')  
            plt.clf()

        if j == 6:
            Test_eth_csv = f'/home/shelley/Desktop/hucares/DL_final2/test_pred_id_{j}_fold{i}.csv'
            df_eth_test = pd.read_csv(Test_eth_csv, index_col=0).astype('float16')
            df_eth_test['test_eth_y_true'] = test_merged[f'Target_{6}']
            df_eth_test.rename(columns = {"0":'test_eth_pred'}, inplace = True)
            df_eth_test=df_eth_test.dropna()
            # print("df_eth_test :  \n") 
            # print(df_eth_test)
            

            test_eth_y_true = list(df_eth_test['test_eth_y_true'])
            test_eth_pred = list(df_eth_test['test_eth_pred'])
            total_eth_true += test_eth_y_true
            total_eth_test += test_eth_pred
            if i == 0:
                avg_eth_test = test_eth_pred

                
            else:
                avg_bit_test = np.sum([avg_bit_test,test_bit_pred], axis=0)

            
            rms_eth = mean_squared_error(test_eth_y_true,test_eth_pred , squared=False)
            pearsonr_eth_score = pearsonr(test_eth_y_true,test_eth_pred )
            corr_eth_score  = np.corrcoef(test_eth_y_true,test_eth_pred )
            print(f'pearsonr score of eth ID={j}_fold{i} is {corr_eth_score}')
            print(f'\corr score of eth ID={j}_fold{i} is {corr_eth_score}')
            print(f'eth (ID=1)  RMSE is {rms_eth}')
                        
            # plot test predicts
            plt.plot(test_eth_y_true, 'r')
            plt.savefig(f'ID={j}_fold{i}_test_eth_y_true.png')  
            plt.clf()
            
            plt.plot(test_eth_pred, 'r')
            plt.savefig(f'ID={j}_fold{i}_test_eth_pred.png')  
            plt.clf()





print('--'*100)
print('')
rms_bit = mean_squared_error(total_bit_true , total_bit_test, squared=False)
pearsonr_bit_score = pearsonr(total_bit_true ,total_bit_test)
corr_bit_score = np.corrcoef(total_bit_true ,total_bit_test)
print(f'pearsonr score of bit ID={j}_fold{i} is {corr_bit_score}')
print(f'corr score of bit ID={j}_fold{i} is {corr_bit_score}')
print(f'bit (ID=1) RMSE is {rms_bit}')
    


rms_eth = mean_squared_error(total_eth_true,total_eth_test , squared=False)
pearsonr_eth_score = pearsonr(total_eth_true,total_eth_test )
corr_eth_score  = np.corrcoef(total_eth_true,total_eth_test )
print(f'pearsonr score of eth ID={j}_fold{i} is {corr_eth_score}')
print(f'\corr score of eth ID={j}_fold{i} is {corr_eth_score}')
print(f'eth (ID=1)  RMSE is {rms_eth}')
                        
        # test_pred_data_process