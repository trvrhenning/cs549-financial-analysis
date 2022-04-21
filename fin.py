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

#check if loaded correctly
data = np.array(readerlist)
print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)
print(data[0,3]) #should be target column

#First implement Logistic Regression

#Set up X and Y, Then , X_train, Y_train, X_test, Y_test 
#Extract features into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18]
print(X_data[0,0]) #check if sliced correctly 

#transpose X 
X_data = X_data.T
print("X_data Shape : {}".format(X_data.shape))

#Extract Ground truth data, array of shape (1, 220510)
#remove title row before Transpose
pure_data = data[1:data.shape[0],:data.shape[1]] #array of (225010, 18)

#transpose for Y
Y_data = pure_data.T

#transfrom one hot encoded target column
Y_data = np.argmax(Y_data, axis=0).reshape((1, Y_data.shape[1]))

#Pick values where financial incidents occured
X_cat0 = X_data[:, np.where(Y_data == 0)[1]]
Y_cat0 = Y_data[:, np.where(Y_data == 0)[1]]
X_cat1 = X_data[:, np.where(Y_data == 1)[1]]
Y_cat1 = Y_data[:, np.where(Y_data == 1)[1]]

print("Y_data shape: {}".format(Y_data.shape)) 

#Convert the label of Y_cat0 to 0, and Y_cat1 to 1
Y_cat0 = np.zeros_like(Y_cat0)
Y_cat1 = np.ones_like(Y_cat1)

print("Y_cat0 shape : {}".format(Y_cat0.shape))
print("Y_cat1 shape : {}".format(Y_cat1.shape))
print("X_cat0 shape : {}".format(X_cat0.shape))
print("X_cat1 shape : {}".format(X_cat1.shape))
