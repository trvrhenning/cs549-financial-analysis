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
#company id columns
print(data[0,0])
#split by company ID or cyclical structure of years

def split_data(data_m):
    #try by company id
    #list of arrays seperated by company
    new_data_m = [ ] 
    #seperated list of a unique company
    bucket = [20]
    company_id = data_m[0,0]
    i = 0 
    headidx = 0
    while (i <  data_m.shape[0]):
        if (company_id == data_m[i,0]):
            i+=1 
        else:
            company_id = data_m[i,0]
            new_data_m.append(data_m[headidx:i,:]) 
            headidx = i
            bucket.clear()
            i+=1
    return new_data_m

#list of numpy arrays seperated by company id
new_data = np.array(split_data(data), dtype = object)
print(f"Check values: {new_data[0].shape}")


#now seperate by max sequence length and we can then convert each sequence into tensors 
#below code will need to be changed 
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

#class LSTM
#class GRU 