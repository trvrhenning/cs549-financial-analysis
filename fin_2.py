#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial data from given csv file

import torch 
import torch.nn as nn
import pandas as pd 

#First Load data from CSV File using pandas this time
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
data_tensor = torch.tensor(raw_data.values)

print (data_tensor.size(dim = 1)) #should be 18
print (data_tensor[0][0].item()) #should be 6 

#Preprocess data 