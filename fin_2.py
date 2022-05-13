#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial incidents from given csv file

import torch 
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
  
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
raw_data = raw_data.fillna(0)
data = np.array(raw_data.values)
data = data[:,1:]

labels = data[:,2]
pos_labels  = np.nonzero(labels)
pos_labels = pos_labels[0]
print(pos_labels.shape)
print(pos_labels[0])

data_tensor = torch.FloatTensor(data)

def split_data(data_m, seq_len, pos_labels):
    seq = [ ]
    for i in pos_labels:
        curr_year = i
        prev_year = curr_year - 1
        count = 0 
        while(data_m[:,1][curr_year] > data_m[:,1][prev_year] and count < seq_len):
            curr_year-=1
            prev_year = curr_year - 1
            count+=1
        if(curr_year == i):
            continue
        seq.append((data_m[curr_year:i,3:17], data_m[i,2]))
    test_size = int(np.round(0.3 * len(seq)))
    train = seq[:-test_size]
    test = seq[-test_size:]
    return train, test 

max_sequence = 5 
train, test = split_data(data_tensor,max_sequence,pos_labels)

input_size = 14
hidden_size = 100 
num_layers = 2 
output_size = 1 
num_epochs = 10 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size 
        self.input_size = input_size
        self.num_layers = num_layers
    
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        #h_cell = h0 and c0 combined 
        self.h_cell = (torch.zeros(self.num_layers,1, self.hidden_size),torch.zeros(self.num_layers,1, self.hidden_size))
        
    def forward(self,x): 
        out, self.h_cell = self.lstm(x.view(len(x),1,-1),self.h_cell)
        output = self.fc(out.view(len(x),-1))
        return output[-1]
       
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size ,num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size 
        self.input_size = input_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size) 
        self.h_cell = torch.zeros(self.num_layers,1, self.hidden_size)
    def forward(self, x):
        out, self.h_cell = self.gru(x.view(len(x),1,-1),self.h_cell)
        output = self.fc(out.view(len(x),-1))
        return output[-1]
    

def train_model(model, model_type, train_data,  num_epochs, print_every = 1000, learning_rate = 0.05):
    model.train()
    print("Training" + model_type + f" model with {num_epochs} epochs:")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for i in range(num_epochs):
        for seq_m, labels in train_data:
            optimizer.zero_grad()
            if (model_type == "LSTM"):
                model.h_cell = (torch.zeros(model.num_layers,1, model.hidden_size),
                                torch.zeros(model.num_layers,1, model.hidden_size))
            else:
                model.h_cell = torch.zeros(model.num_layers,1, model.hidden_size)
            y_pred = model(seq_m)
            loss = criterion(y_pred,labels)
            loss.backward()
            optimizer.step()
            
        if(i % print_every == 0 and i > 0):
            print(f"Epoch:{i} loss: {loss.item():10.8f}")
        print(f"Epoch {i} loss : {loss.item():10.10f}")

lstm_model = LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size)
train_model(lstm_model, "LSTM", train, num_epochs)

gru_model = GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = output_size)
train_model(gru_model, "GRU", train, num_epochs)