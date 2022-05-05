#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial incidents from given csv file

import torch 
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#First Load data from CSV File using pandas this time for ease of use with pytorch
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')

#Create a numpy array and extract X and Y 
data = np.array(raw_data.values)
#slice unneeded column
data = data[:,1:data.shape[1]]

#split data by company ID
def split_data(data_m):
    new_data_m = [ ]
    bucket = [20] 
    company_id = data_m[0,0]
    idx = 0 
    headidx = 0
    while (idx <  data_m.shape[0]):
        if (company_id == data_m[idx,0]):
            idx+=1 
        else:
            company_id = data_m[idx,0]
            new_data_m.append(data_m[headidx:idx,:]) 
            headidx = idx
            bucket.clear()
            idx+=1 
    #returns list of numpy arrays seperated by company id
    return new_data_m

#list of numpy arrays seperated by company id
new_data = np.array(split_data(data), dtype = object)
test_size = int(np.round(0.3*new_data.shape[0]))
train_data = new_data[:(new_data.shape[0] - test_size)]
test_data = new_data[(new_data.shape[0] - test_size):new_data.shape[0]]
print(f"Check train data shape: {train_data.shape}")

#train_data[i].shape and test_data[i].shape = (samples_each_company, features)
#need to slice each companys data ie. train_data[i] by a max sequence length, but smaller if less length than max sequence
#features, 1-3 are company id year and target val, thus we dont wnat those in calculations for training

#below code will need to be changed, but process of converting to tensors should be similar. 

X_data = np.concatenate((data[:,1:3],data[:,4:18]), axis = 1) 
Y_data = data[:,3]
print()
print("Data shape : {} ".format(data.shape))
print("X_data shape : {}".format(X_data.shape))
print("Y_data shape : {}".format(Y_data.shape))

#Create X, Y Train and X, Y Test before converting to tensors
Xtr, Xte, Ytr, Yte = train_test_split(X_data, Y_data, test_size = 0.35, shuffle = True)

#Convert to tensor dataset and then use dataloader
X_train = torch.tensor(Xtr)
Y_train = torch.tensor(Ytr)
X_test = torch.tensor(Xte)
Y_test = torch.tensor(Yte)

train_set = TensorDataset(X_train, Y_train)
test_set = TensorDataset(X_test, Y_test) 

#Load with dataloader here, need to get batch size and other params as well
train = DataLoader(train_set, batch_size = 64, shuffle = False)
test = DataLoader(test_set, batch_size = 64, shuffle = False)

print("X_train shape: {}".format(X_train.size()))
print("X_test shape: {}".format(X_test.size()))
print("Y_train shape: {}".format(Y_train.size()))
print("Y_test shape: {}".format(Y_test.size()))

#Create RNN classes

input_size = 14
hidden_size = 100 
num_layers = 2 
output_size = 1 
num_epochs = 100 

#class LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size)
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size 
    self input_size = input_size
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self,x): 
        #fill in code here 
        pass
    
#class GRU

class GRU(nn.Module):
    def __init__(self, input, hidden_size,num_layers, output_size)
    super(GRU, self).__init__()
    self.hidden_size = hidden_size 
    self input_size = input_size
    
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        #fill in code here
        pass 
        