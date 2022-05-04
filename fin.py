#CS 549 Final Project, Financial Analysis, Group 23
#This python file implements 3 non sequential machine learning models on financial incidents from given csv file

import csv
import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

#First Load data from CSV File 
raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
reader  = csv.reader(raw_data, delimiter = ',' , quoting = csv.QUOTE_NONE) 

#Create an array of the data from a list set empty feature values to 0
readerlist = list(reader)
data = np.array(readerlist)
data[data == ''] = 0

#Check if data is extracted correctly
print()
print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)

#Set up X and Y :
#Extract features and samples into an array of size (220510,14)
X_data = data[1:data.shape[0],4:18] 
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

#Scale data 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Under sample data to even out imbalance of positive and negative outcomes
#Use different UnderSampling methods sampling_strategy rates for best model outcomes!
rus = RandomUnderSampler(sampling_strategy = 0.6, random_state = 42)
X_train_resample, Y_train_resample = rus.fit_resample(X_train, Y_train)
print("X_train resampled shape: {}".format(X_train_resample.shape))
print("Y_train resampled shape: {}".format(Y_train_resample.shape))


#Implement 3 Sci-kit Models and evaluate their performances Below 

start = time.time()

#1. Logistic Regression
#Initialiaze model and make predictions on test data 
l_reg = LogisticRegression() 
l_reg.fit(X_train_resample,Y_train_resample)
y_pred = l_reg.predict(X_test)

#Evalaute Logistic Regression Model 
cm = metrics.confusion_matrix(Y_test, y_pred) 
accuracy = l_reg.score(X_test, Y_test) 
print("Confusion Matrix for Logistic Regression Model : \n{}".format(cm)) 
print("Logistic Regression Model Accuracy : {0:.2%}".format(accuracy))

#2. SVM , this one may take some time to run
svm_model = SVC()
svm_model.fit(X_train_resample,Y_train_resample)
y_pred_1 = svm_model.predict(X_test)

#Evaluate SVM model
cm_1 = metrics.confusion_matrix(Y_test, y_pred_1)
accuracy_1 = svm_model.score(X_test, Y_test)
print("Confusion Matrix for SVM Model : \n{}".format(cm_1))
print("SVM Model Accuracy : {0:.2%}".format(accuracy_1))

#3. Feed forward Neural Network
mlp_model = MLPClassifier(max_iter = 1000) #Hidden layer default at (100,)
mlp_model.fit(X_train_resample,Y_train_resample)
y_pred_2 = mlp_model.predict(X_test)

#Evaluate Neural Network Model  
cm_2 = metrics.confusion_matrix(Y_test, y_pred_2) 
accuracy_2 = mlp_model.score(X_test, Y_test)
print("Confusion Matrix for Neural Network Model : \n{}".format(cm_2))
print("Neural Network Model Accuracy : {0:.2%}".format(accuracy_2))   

end = time.time()

print("Total time for all 3 models to run : {:.2f}".format(end - start)) 