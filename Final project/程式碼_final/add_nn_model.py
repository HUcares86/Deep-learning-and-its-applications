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

from sklearn.metrics import mean_squared_error
lags = [60,300,900]
lags_interval = [[5,10],[11,20],[21,40],[41,60],[61,80],[81,100],[101,200]]
id_list = [1,6]
# train data input ---------------------------------------------------
TRAIN_CSV = "/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/train_feat.csv"
training_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/train_feat_y_true.csv'


df_train = pd.read_csv(TRAIN_CSV, index_col=0)
# print('--'*50)
# print(df_train)
# print('')

df_train_y_true = pd.read_csv(training_y_true_csv, index_col=0)
# print('--'*50)
# print(df_train_y_true)
# print('')


# testing data input -----------------------------------------------------------------
Test_csv = "/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/test_feat.csv"
Test_y_true_csv = '/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/test_feat_y_true.csv'
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
            train_bit_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/train_pred_id_{j}_fold{i}.csv'
            df_bit_train = pd.read_csv(train_bit_csv, index_col=0).astype('float16')
            df_train[f'train_pred_id_{j}_fold{i}'] = df_bit_train
            
            test_bit_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/test_pred_id_{j}_fold{i}.csv'
            df_bit_test = pd.read_csv(test_bit_csv, index_col=0).astype('float16')
            df_test[f'test_pred_id_{j}_fold{i}'] = df_bit_test


        if j == 6:
            train_eth_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/train_pred_id_{j}_fold{i}.csv'
            df_eth_train = pd.read_csv(train_eth_csv, index_col=0).astype('float16')
            df_train[f'train_pred_id_{j}_fold{i}'] = df_eth_train
            
            test_eth_csv = f'/home/shelley/Desktop/hucares/DL_final2/2022_0504_to_now_2/feature_plus_2_model/test_pred_id_{j}_fold{i}.csv'
            df_eth_test = pd.read_csv(test_eth_csv, index_col=0).astype('float16')
            df_test[f'test_pred_id_{j}_fold{i}'] = df_eth_test


df_test_total = df_test
df_test_total['Target_1'] = df_test_y_true['Target_1']
df_test_total['Target_6'] = df_test_y_true['Target_6']
df_test_total = df_test_total.dropna()
print("Total Null Target Rows = " ,df_test_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_test_total["Target_1"].isnull().sum()*100 / df_test_total.shape[0] ))
print("--------"*15)

df_train_total = df_train
df_train_total['Target_1'] = df_train_y_true['Target_1']
df_train_total['Target_6'] = df_train_y_true['Target_6']
df_train_total = df_train_total.dropna()
print("Total Null Target Rows = " ,df_train_total["Target_1"].isnull().sum())
print("Percentage of NUll rows in Training Data = {:.2f}%".format(df_train_total["Target_1"].isnull().sum()*100 /df_train_total.shape[0] ))
print("--------"*15)

df_test_y_true = pd.DataFrame()
df_test_y_true['Target_1'] = df_test_total['Target_1'] 
df_test_y_true['Target_6'] = df_test_total['Target_6'] 
df_test = df_test_total.drop('Target_1',axis =1)
df_test = df_test_total.drop('Target_6',axis =1)

df_train_y_true = pd.DataFrame()
df_train_y_true['Target_1'] = df_train_total['Target_1'] 
df_train_y_true['Target_6'] = df_train_total['Target_6'] 
df_train = df_train_total.drop('Target_1',axis =1)
df_train = df_train_total.drop('Target_6',axis =1)



# training target (y_true)-------------------------------------------------------------
target_training = df_train_y_true
target_training = torch.tensor(target_training.to_numpy().astype(np.float32))
target_training = target_training.view(target_training.shape[0], 2)
# print('target_training', '--'*50)
# print(target_training)
# print('')

# training feature (x's)-------------------------------------------------------------
feature_training = torch.from_numpy(df_train.to_numpy().astype(np.float32))
# print('feature_training', '--'*50)
# print(feature_training)
# print('')

# testing target (y_true)-------------------------------------------------------------
target_testing = df_test_y_true
target_testing = torch.tensor(target_testing.to_numpy().astype(np.float32))
target_testing = target_testing.view(target_testing.shape[0], 2)
# print('target', '--'*50)
# print(target_testing)
# print('')
# print(len(target_testing ))

# testing feature (x's)-------------------------------------------------------------
feature_testing = torch.from_numpy(df_test.to_numpy().astype(np.float32))
# print('feature_testing', '--'*50)
# print(feature_testing)
# print('')


df_test = df_test.values
df_test  = df_test.reshape((df_test.shape[0], 1, df_test.shape[1]))
df_train = df_train.values
df_train = df_train.reshape((df_train.shape[0], 1, df_train.shape[1]))
df_test_y_true2 = df_test_y_true
df_test_y_true = df_test_y_true.values
df_train_y_true = df_train_y_true.values
print ("df_test.shape" , df_test.shape) 
print ("df_test_y_true.shape" ,df_test_y_true.shape) 
print ("df_train.shape" , df_train.shape) 
print ("df_train_y_true .shape" , df_train_y_true .shape) 

print(df_test[0])
print(df_test_y_true[0])
print(df_train_y_true[0])
print(df_train[0])

split = round(df_train.shape[0]*0.95)
train_X , train_y = df_train[:split, :] ,  df_train_y_true[:split, :]
test_X , test_y = df_train[split:, :] ,  df_train_y_true[split:, :]

n_features = train_X.shape[2]
print(n_features)
n_steps_in, n_steps_out = 1 , 2
#optimizer learning rate
opt = keras.optimizers.legacy.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-09) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-09

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mse' , optimizer=opt , metrics=['mse'])

history = model.fit(train_X , train_y , epochs=5 , steps_per_epoch=50 , verbose=1 ,validation_data=(test_X, test_y),shuffle=False)

def prep_data(df_test, start , end, last):
    #prepare test data X
    dataset_test = df_test
    # print(dataset_test)
    dataset_test_X = dataset_test[start:end, :]
    print(dataset_test_X.shape)
    # test_X_new = dataset_test_X.reshape(1, dataset_test_X.shape[0] , dataset_test_X.shape[1])

  # predictions
    y_pred = model.predict(dataset_test_X)
    print(y_pred.shape)
    # print(y_pred)
    # y_pred_inv = scaler1.inverse_transform(y_pred)
    # print(y_pred_inv)
    y_pred_inv = y_pred.reshape(n_steps_out,1)
    # print(y_pred_inv)
    # y_pred_inv = y_pred_inv[:,0]
    
    return y_pred

ans = []
for i in range(0,df_test.shape[0],1):
    print("i: ",i)
    start = i
    end = start + n_steps_in 
    last = end + n_steps_out 
    dataset_test_X = df_test[start:end, :]
    print(dataset_test_X.shape)
    y_pred = model.predict(dataset_test_X)
    print(y_pred)
    print(y_pred.shape)
    ans.append(y_pred)

# y_pred = model.predict(df_test)
print(y_pred_inv)

bit_pred = []
eth_pred = []
for i in range(len(y_pred_inv)):
    bit_pred += [y_pred_inv[i][0]]
    eth_pred += [y_pred_inv[i][1]]
    
all_final_test_predict = pd.DataFrame()
all_final_test_predict['bit_pred'] = bit_pred
all_final_test_predict['eth_pred'] = eth_pred

print(all_final_test_predict)
all_final_test_predict.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/','all_final_test_predict.csv')) #save to file   

plt.plot(bit_pred , 'r')
plt.savefig(f'bit_pred.png')  
plt.clf()

plt.plot(eth_pred , 'b')
plt.savefig(f'eth_pred.png')  
plt.clf()

bit_y_true = list(df_test_y_true2['Target_1'])
eth_y_true = list(df_test_y_true2['Target_6'])

all_final_test_predict['bit_y_true'] = df_test_y_true2['Target_1']
all_final_test_predict['eth_y_true'] = df_test_y_true2['Target_6']
print(all_final_test_predict)
all_final_test_predict.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/','all_final_test_predict.csv')) #save to file   


plt.plot(bit_y_true , 'r')
plt.savefig(f'bit_y_true.png')  
plt.clf()

plt.plot(eth_y_true , 'b')
plt.savefig(f'eth_y_true.png')  
plt.clf()

rms_bit = mean_squared_error(bit_y_true,bit_pred , squared=False)
pearsonr_bit_score = pearsonr(bit_y_true,bit_pred )
corr_bit_score  = np.corrcoef(bit_y_true, bit_pred)
print(f'pearsonr score of bit ID={j} is {corr_bit_score}')
print(f'\corr score of bit ID={j} is {corr_bit_score}')
print(f'bit (ID=1)  RMSE is {rms_bit}')

rms_eth = mean_squared_error(eth_y_true,eth_pred , squared=False)
pearsonr_eth_score = pearsonr(eth_y_true,eth_pred )
corr_eth_score  = np.corrcoef(eth_y_true,eth_pred )
print(f'pearsonr score of eth ID={6} is {corr_eth_score}')
print(f'\corr score of eth ID={6} is {corr_eth_score}')
print(f'eth (ID=6)  RMSE is {rms_eth}')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1) model
class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_dim):        #
        super(LinearRegression, self).__init__()
        
        # define layers
        self.linear1 = nn.Linear(input_dim, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)

        self.linear11 = nn.Linear(hidden_size4, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)

        out = self.linear2(out)
        out = F.relu(out)

        out = self.linear3(out)
        out = F.relu(out)

        out = self.linear4(out)
        out = F.relu(out)

        out = self.linear11(out)

        return out


model = LinearRegression(n_features, 60, 60, 60, 60 , 2)

ans = torch.empty(len(target_testing),2)
print(ans)
yb = model(feature_training[0:5])
print(yb)
# ans[0] = yb
# print(ans)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3) training loop
epochs = 500
batch_size = 100
#Mean Sqaured Error Loss
loss_fn = torch.nn.MSELoss()
# mse(y_true, y_pred).numpy()
for epoch in range(epochs):
  for i in range(int(len(feature_training)/batch_size)):
      # forward pass and loss 
      y_predicted = model(feature_training[100*i:100*(i+1)])
    #   print("y_predicted")
    #   print(y_predicted)
      print("target_training[100*i:100*(i+1)]")
      print(feature_training[100*i:100*(i+1)])
      loss = loss_fn(y_predicted, target_training[100*i:100*(i+1)])
      print(loss)
      

      # backward pass
      loss.backward()

      # update
      optimizer.step()
      
      # init optimizer
      optimizer.zero_grad()

  if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

# # show in image
# predicted = model(feature).detach().numpy()
# plt.show()


FILE = 'test1_4.pt'
torch.save(model, FILE)

ans = torch.empty(len(target_testing),2)
for i in range(len(target_testing)):
  yb = model(feature_testing [0])
  ans[i] = yb


print(ans)

import pandas as pd
import os.path
ans_np = ans.detach().numpy() #convert to Numpy array
print(ans_np)




df_ans = pd.DataFrame(ans_np) #convert to a dataframe

df_ans.to_csv(os.path.join(f'/home/shelley/Desktop/hucares/DL_final2/',f'df_ans.csv')) #save to file  


from scipy.stats.stats import pearsonr   
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

target_training

rms_bit = mean_squared_error(test_bit_y_true,test_bit_pred , squared=False)
pearsonr_bit_score = pearsonr(test_bit_y_true,test_bit_pred )
corr_bit_score  = np.corrcoef(test_bit_y_true,test_bit_pred )
print(f'pearsonr score of bit ID={1} is {corr_bit_score}')
print(f'\corr score of bit ID={1} is {corr_bit_score}')
print(f'bit (ID=1)  RMSE is {rms_eth}')

rms_eth = mean_squared_error(test_eth_y_true,test_eth_pred , squared=False)
pearsonr_eth_score = pearsonr(test_eth_y_true,test_eth_pred )
corr_eth_score  = np.corrcoef(test_eth_y_true,test_eth_pred )
print(f'pearsonr score of eth ID={6} is {corr_eth_score}')
print(f'\corr score of eth ID={6} is {corr_eth_score}')
print(f'eth (ID=1)  RMSE is {rms_eth}')