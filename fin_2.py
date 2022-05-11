#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial incidents from given csv file

import torch 
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#First Load data 
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
data = np.array(raw_data.values)
data = data[:,1:data.shape[1]]

#Splits data by company Id and then by sequence length
def split_data(data_m, seq_len):
    new_data_m = [ ]
    y_seq = [ ]
    x_seq = [ ]
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
    for i in range(len(new_data_m)):
        if(new_data_m[i].shape[0] < seq_len):
            x_seq.append(new_data_m[i][:,3:17])
            y_seq.append(new_data_m[i][:,2])
        else:
            x_seq.append(new_data_m[i][-seq_len:,3:17])
            y_seq.append(new_data_m[i][-seq_len:,2])

    x_data = np.array(x_seq , dtype = object)
    y_data = np.array(y_seq , dtype = object)
    test_size = int(np.round(0.3 * x_data.shape[0]))
    x_train = x_data[:-test_size]
    y_train = y_data[:-test_size]
    x_test = x_data[(x_data.shape[0] - test_size):] 
    y_test = y_data[(y_data.shape[0] - test_size):]
    
    return x_train, y_train, x_test, y_test

max_sequence = 5 
X_train, Y_train, X_test, Y_test = split_data(data, max_sequence)
print(f"Check train data shape: {Y_train.shape}")

batch_size = 64
input_size = 14
hidden_size = 100 
num_layers = 2 
output_size = 1 
num_epochs = 100 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size 
        self.input_size = input_size
    
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self,x): 
        #fill in code here 
        pass
    
class GRU(nn.Module):
    def __init__(self, input, hidden_size,num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size 
        self.input_size = input_size
    
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        #fill in code here
        pass 
        