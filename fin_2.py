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

#split data by company ID and then seperate into sequences, may need to remove last row for test data
def split_data(data_m, seq_len):
    new_data_m = [ ]
    sequences = [ ]
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
           # sequences.append(new_data_m[i])
            x_seq.append(new_data_m[i][:,4:18])
            y_seq.append(new_data_m[i][:,3])
        else:
           # sequences.append(new_data_m[i][-seq_len:,:])
            x_seq.append(new_data_m[i][:,4:18])
            y_seq.append(new_data_m[i][:,3])

    x_data = np.array(x_seq , dtype = object)
    y_data = np.array(y_seq , dtype = object)
    test_size = int(np.round(0.3 * x_data.shape[0]))
    x_train = x_data[:(x_data.shape[0] - test_size)]
    y_train = y_data[:(y_data.shape[0] - test_size)]
    x_test = x_data[(x_data.shape[0] - test_size):x_data.shape[0]] 
    y_test = y_data[(y_data.shape[0] - test_size):y_data.shape[0]]
    
    return x_train, y_train, x_test, y_test

#list of numpy arrays seperated by company id each in length of chosen sequence lenght
#new_data = np.array(split_data(data, 5), dtype = object)
#test_size = int(np.round(0.3 * new_data.shape[0]))
#train_data = new_data[:(new_data.shape[0] - test_size)]
#test_data = new_data[(new_data.shape[0] - test_size):new_data.shape[0]]
#print(f"Check train data shape: {train_data.shape}") #size should be amount of sequences

#train_data[i].shape and test_data[i].shape = (samples_each_company(size of sequence length), features) 
#features, 1-3 are company id year and target val, thus we dont wnat those in calculations for training

X_train, Y_train, X_test, Y_test = split_data(data, 5)
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
        