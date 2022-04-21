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

data = np.array(readerlist)
print(data.shape) #225011, 18 
print(data[0,3]) #should be target column