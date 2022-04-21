#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial incidents from given csv file

import torch 
import torch.nn as nn
import pandas as pd 

#First Load data from CSV File using pandas this time for ease of use with pytorch
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
rt_data = torch.tensor(raw_data.values)

#Check if values were loaded correctly
print()
print("Checking if rt_data is properly loaded into tensor: \n")
print("This value should be 18 : {}".format(rt_data.size(dim = 1))) #should be 18
print("This value should be 6.0 : {}".format(rt_data[0][0].item())) #should be 6 
print()

#Slice first unneeded row value column from csv data
data = rt_data[:rt_data.size(dim = 0),1:rt_data.size(dim =1)] #tensor of size ([225010,17]) 

#Check dimensions and values
print(data.size()) 
print("This value should be 1004 : {}".format(data[0][0]))
print("This value should be 2000 : {}".format(data[0][1]))

#Preprocess data tensor further below if necessary 