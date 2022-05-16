#CS 549 Final Project, Financial Analysis
#This python file implements 3 non sequential machine learning models on financial data from a given csv file

import csv
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def process_data():
    
    raw_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
    reader  = csv.reader(raw_data, delimiter = ',' , quoting = csv.QUOTE_NONE) 
    
    #Create an array of the data from a list set empty feature values to 0
    readerlist = list(reader)
    raw_data.close()
    data = np.array(readerlist)
    data[data == ''] = 0

    #Check if data is extracted correctly
    print()
    print("Checking extraction shape: {}".format(data.shape)) #array of (225011, 18)

    #Set up X and Y :
    X_data = data[1:data.shape[0],4:18] 
    print("X_data Shape: {}".format(X_data.shape))
    Y_data = data[1:data.shape[0],3] 
    print("Y_data Shape: {}".format(Y_data.shape))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.35, shuffle = True)

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("Y_train shape: {}".format(Y_train.shape))
    print("Y_test shape: {}".format(Y_test.shape))

    #Scale data 
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Under sample data to even out imbalance of positive and negative outcomes
    #Use different sampling_strategy rates for best model outcomes!
    rus = RandomUnderSampler(sampling_strategy = 0.6, random_state = 42)
    X_train_resample, Y_train_resample = rus.fit_resample(X_train, Y_train)
    print("X_train resampled shape: {}".format(X_train_resample.shape))
    print("Y_train resampled shape: {}".format(Y_train_resample.shape))
    print("The data: ", data[0:6])

    return X_train_resample, Y_train_resample, X_test, Y_test

#1. Logistic Regression
def log_reg_m(X_train_m,Y_train_m,X_test_m,Y_test_m):
    l_reg = LogisticRegression() 
    l_reg.fit(X_train_m,Y_train_m)
    y_pred = l_reg.predict(X_test_m)

    #Evalaute Logistic Regression Model here
    cm = metrics.confusion_matrix(Y_test_m, y_pred) 
    accuracy = l_reg.score(X_test_m, Y_test_m) 
    print("Confusion Matrix for Logistic Regression Model : \n{}".format(cm)) 
    print("Logistic Regression Model Accuracy : {0:.2%}".format(accuracy))
    print("Logistic Regression Model F1 Score: ", f1_score(Y_test_m, y_pred, average=None))
    Y_test_m2 = '1' <= Y_test_m
    y_pred_proba = l_reg.predict_proba(X_test_m)[::,1]
    fpr, tpr, _ = roc_curve(Y_test_m2, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.show()

#2. SVM , this one may take some time to run
def svm_m(X_train_m,Y_train_m,X_test_m,Y_test_m):
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_m,Y_train_m)
    y_pred_1 = svm_model.predict(X_test_m)
    print(y_pred_1[0:5])

    #Evaluate SVM model here 
    cm_1 = metrics.confusion_matrix(Y_test_m, y_pred_1)
    accuracy_1 = svm_model.score(X_test_m, Y_test_m)
    print("Confusion Matrix for SVM Model : \n{}".format(cm_1))
    print("SVM Model Accuracy : {0:.2%}".format(accuracy_1))
    print("SWM Model F1 Score: ", f1_score(Y_test_m, y_pred_1, average=None))
    Y_test_m2 = '1' <= Y_test_m
    #y_pred_proba = svm_model.predict_proba(X_test_m)[::,1]
    y_pred_proba = svm_model.predict_proba(X_test_m)[:,1]
    fpr, tpr, _ = roc_curve(Y_test_m2, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.title("SVM Model ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

#3. Feed forward Neural Network
def nnet_m(X_train_m, Y_train_m, X_test_m, Y_test_m):
    mlp_model = MLPClassifier(max_iter = 1000) #Hidden layer default at (100,)
    mlp_model.fit(X_train_m,Y_train_m)
    y_pred_2 = mlp_model.predict(X_test_m)

    #Evaluate Neural Network Model here   
    cm_2 = metrics.confusion_matrix(Y_test_m, y_pred_2) 
    accuracy_2 = mlp_model.score(X_test_m, Y_test_m)
    print("Confusion Matrix for Neural Network Model : \n{}".format(cm_2))
    print("Neural Network Model Accuracy : {0:.2%}".format(accuracy_2))
    print("Neural Network Model F1 Score: ", f1_score(Y_test_m, y_pred_2, average=None))
    Y_test_m2 = '1' <= Y_test_m
    y_pred_proba = mlp_model.predict_proba(X_test_m)[:,1]
    fpr, tpr, _ = roc_curve(Y_test_m2, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.title("Neural Network Model ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = process_data()
    #p1 = mp.Process(target = log_reg_m, args = (X_train, Y_train, X_test, Y_test))
    #p2 = mp.Process(target = svm_m, args = (X_train, Y_train, X_test, Y_test))
    p3 = mp.Process(target = nnet_m, args = (X_train, Y_train, X_test, Y_test))
    
    start = time.time()
    #p1.start()
    #p2.start()
    p3.start()
    
    #p1.join()
    #p2.join()
    p3.join()
    end = time.time()
    
    print("Finished in {:.2f} seconds ".format(end - start))     