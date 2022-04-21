#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 2 sequential machine learning models on financial incidents from given csv file

import torch 
import torch.nn as nn
import pandas as pd 

#First Load data from CSV File using pandas this time for ease of use with pytorch
raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
data_tensor = torch.tensor(raw_data.values)

print()
print("Checking if data is properly loaded into tensor: \n")
print ("This value should be 18 : {}".format(data_tensor.size(dim = 1))) #should be 18
print ("This value should be 6.0 : {}".format(data_tensor[0][0].item())) #should be 6 
print()


#Preprocess any data if necessary here