#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 3 non sequential machine learning models on financial data from a given csv file

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt 

#First Load data from CSV File 
raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
reader  = csv.reader(raw_data, delimiter = ',' , quoting = csv.QUOTE_NONE) 

#Create an array of the data from a list
readerlist = list(reader)
data = np.array(readerlist)

#Try Loading with pandas instead - comment out above code before using: 
#raw_data = pd.read_csv('targetfirm_prediction_dataset_small.csv')
#raw_data = raw_data.fillna(0)
#data = np.array(raw_data) 

#Check if data is extracted correctly
print()
print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)
print("Checking if values are correct ")
print("This value should be 0.0 : {}".format(data[0,3])) #should be target column


#Set up X and Y 
#Extract features and samples into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18] #dont need to slice first row 

#Check if X was sliced correctly
print("This value should be 701.85399: {}".format(X_data[0,0]))
print("X_data Shape: {}".format(X_data.shape))

#Extract target values into array of size (220510,)
Y_data = data[1:data.shape[0],3] 

print("Y_data Shape: {}".format(Y_data.shape))

#Create X, Y Train and X, Y Test 
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.35, shuffle = True)

#Check dimensions of train and test sets 
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("Y_test shape: {}".format(Y_test.shape))

#Code below fixes the empty string issue 
X_train[X_train == ''] = 0
X_test[X_test == ''] = 0
X_train = np.array(X_train, dtype = float)
X_test = np.array(X_test, dtype = float)
Y_train = np.array(Y_train, dtype = float)
Y_test = np.array(Y_test, dtype = float)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

print("Checking arrays are not all zeros : {}".format(np.count_nonzero(X_test)))
#Data Preprocessed!!

#Implement 3 Sci-kit Models Below

#Still unsure about performance or correctness of these models!! No true positive in confusion matrices!!

#1. Logistic Regression
#Initialiaze scikit model 
l_reg = LogisticRegression(max_iter = 1000) 
#Fit data to our model and make predictions
l_reg.fit(X_train_transformed,Y_train)

y_pred = l_reg.predict(X_test_transformed)

#Evalaute Logistic Regression Model 
cm = metrics.confusion_matrix(Y_test, y_pred) 
accuracy = (cm[0,0] + cm[1,1])/len(Y_test)
print("Confusion Matrix for Logistic Regression Model : \n{}".format(cm)) 
print("Logistic Regression Model Accuracy : {0:.2%}".format(accuracy))

#2. SVM 
#Initialize scikit model 
svm_model = SVC(C=1)

#Fit data to our model and make predictions
svm_model.fit(X_train_transformed,Y_train)
y_pred_1 = svm_model.predict(X_test_transformed)

#Evaluate SVM model
cm_1 = metrics.confusion_matrix(Y_test, y_pred_1)
accuracy_1 = (cm_1[0,0] + cm_1[1,1])/len(Y_test)
print("Confusion Matrix for SVM Model : \n{}".format(cm_1))
print("SVM Model Accuracy : {0:.2%}".format(accuracy_1))

#3. Feed forward Nueral Network
#Initialize scikit model
mlp_model = MLPClassifier()

#Fit data to our model and make predictions
mlp_model.fit(X_train_transformed, Y_train)
y_pred_2 = mlp_model.predict(X_test_transformed)

#Evaluate Nueral Network Model  
cm_2 = metrics.confusion_matrix(Y_test, y_pred_2) #Hidden layer default at (100,)
accuracy_2 = (cm_2[0,0] + cm_2[1,1])/len(Y_test)
print("Confusion Matrix for Neural Network Model : \n{}".format(cm_2))
print("Neural Network Model Accuracy : {0:.2%}".format(accuracy_2))