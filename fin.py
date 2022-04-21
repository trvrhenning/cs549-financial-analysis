#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 3 non sequential machine learning models on financial data from a given csv file

import csv
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt 

#First Load data from CSV File 
raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
reader  = csv.reader(raw_data, delimiter = ',' , quoting  = csv.QUOTE_NONE) 

#Create an array of the data from a list
readerlist = list(reader)
data = np.array(readerlist) 

#Check if data is extracted correctly
print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)
print(data[0,3]) #should be target column

#Set up X and Y 
#Extract features and samples into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18]
print(X_data[0,0]) #check if sliced correctly 

print("X_data Shape : {}".format(X_data.shape))

#Extract target values into array of size (220510,)
Y_data = data[1:data.shape[0],3] 

print("Y_data Shape : {}".format(Y_data.shape))

#Implement Sci-kit Models Below

#Create X, Y Train and X, Y Test 

