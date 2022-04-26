#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 3 non sequential machine learning models on financial data from a given csv file

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
import matplotlib.pyplot as plt 

#First Load data from CSV File 
raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
reader  = csv.reader(raw_data, delimiter = ',' , quoting  = csv.QUOTE_NONE) 

#Create an array of the data from a list
readerlist = list(reader)
data = np.array(readerlist) 

#Check if data is extracted correctly
print()
print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)
print("Checking if values are correct ")
print("This value should be target: {}".format(data[0,3])) #should be target column


#Set up X and Y 
#Extract features and samples into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18]

#Check if X was sliced correctly
print("This value should be 701.85399: {}".format(X_data[0,0]))
print("X_data Shape: {}".format(X_data.shape))

#Extract target values into array of size (220510,)
Y_data = data[1:data.shape[0],3] 

print("Y_data Shape: {}".format(Y_data.shape))

#Create X, Y Train and X, Y Test 
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2)

#Check dimensions of train and test sets 
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("Y_test shape: {}".format(Y_test.shape))

#Code below fixes the empty string issue 
X_train[ X_train == ''] = 0
X_test[X_test == ''] = 0
X_train = np.array(X_train, dtype = float)
X_test = np.array(X_test, dtype = float)
Y_train = np.array(Y_train, dtype = float)
Y_test = np.array(Y_test, dtype = float)


#Implement 3 Sci-kit Models Below '

#1. Logistic Regression
#Initialiaze scikit model 
l_reg = LogisticRegression(max_iter = 70000) 

#Fit the model to our data 
l_reg.fit(X_train, Y_train)

#Make predictions
y_pred = l_reg.predict(X_test)

#Create confusion matrix to evaluate the model 
cm  = metrics.confusion_matrix(Y_test, y_pred) 

#Still unsure about performance or correctness of this model!! No true positive in matrix... 

print(cm)