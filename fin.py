#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 3 non sequential machine learning models on financial data from a given csv file

import csv
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt 

#First Load data from CSV File 
raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
reader  = csv.reader(raw_data, delimiter = ',' , quoting  = csv.QUOTE_NONE)
readerlist = list(reader)

#get dimmensions
rowcount = len(readerlist)
colcount = len(readerlist[0])

#check if loaded correctly
data = np.array(readerlist)
print(data.shape) #225011, 18 
print(data[0,3]) #should be target column

#LOGISTIC REGRESSION 
#First set up X and Y, Then , X_train, Y_train, X_test, Y_test 

#Extract features into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18]
print(X_data[0,0]) #check if sliced correctly 
print(X_data.shape)

#Extract Ground truth data, array of shape (1, 220510)
Y_data = data[1:data.shape[0],3] 


Y_data = Y_data.reshape((1, Y_data.shape[0]))
print(Y_data.shape) 


#Pick values where financial incidents occured
X_cat0 = X_data[np.where(Y_data == 0)]
Y_cat0 = Y_data[np.where(Y_data == 0)]
X_cat1 = X_data[np.where(Y_data == 1)]
Y_cat1 = Y_data[np.where(Y_data == 1)] 

# Convert the label of Y_cat0 to 0, and Y_cat1 to 1
Y_cat0 = np.zeros_like(Y_cat0)
Y_cat1 = np.ones_like(Y_cat1)

print(Y_cat1.shape)
