# CS 549 Final Project-Financial Incident Prediction Using Machine Learning:
Contributors: 
Trevor Henning, trvrhenning@gmail.com 
Omid Hedayatnia, omid.hedayat7@gmail.com

Given a data set this program will make predictions of financial incidents (specifically company acquisition) using multiple different machine learning models. 
## Non sequential models 
In the first python file, fin.py,  we tested and evaluated 3 different non sequential models on our data and used multiprocessing to speed up training and evaluation(speed change can be negligible depending on device). The models used are scikit learn's
logistic regression, SVC(svm, support vector machine), and MLP(neural network) models.   
## Sequential models 
In the second python file, fin_2.py, we tested and evaluated 2 different sequential models using pytorch, these being LSTM and GRU style RNNS. 
## INSTRUCTIONS and DEPENDENCIES: 
Before running this program it is required to have a few different python packages.
If you dont have the following libraries, please install before attempting to run: 
Numpy, Sklearn, Pytorch, Pandas, Matplotlib, Imbalanced Learn. 
To install following packages: 
1. pip install numpy 
2. pip install scikit-learn
3. pip install torch
4. pip install pandas
5. pip install matplotlib
6. pip install imbalanced-learn

TO RUN THE PROGRAMS:
1. From the command line cd into the directory where you stored the python files and data from this repository. 
2. To run fin.py, type: python3 fin.py
3. To run fin_2.py type: python3 fin_2.py
Note: these programs may take a few minutes to run! 
