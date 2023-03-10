# -*- coding: utf-8 -*-
"""Random forest_wine prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V5VghYx0SrrcdIqXdbF1-N9_2TOUJysi

Mount Drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""Import Libraries"""

import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd

"""Load Data"""

data = pd.read_csv("/content/drive/MyDrive/AI_data/Winequality_dataset_final.csv")
data.head()

"""# Min max values"""

fixed_min = data['fixed acidity'].min()
fixed_max = data['fixed acidity'].max()
volatile_min = data['volatile acidity'].min()
volatile_max = data['volatile acidity'].max()
citric_min = data['citric acid'].min()
citric_max = data['citric acid'].max()
residual_min = data['residual sugar'].min()
residual_max = data['residual sugar'].max()
chlorides_min = data['chlorides'].min()
chlorides_max = data['chlorides'].max()
free_min = data['free sulfur dioxide'].min()
free_max = data['free sulfur dioxide'].max()
total_min = data['total sulfur dioxide'].min()
total_max = data['total sulfur dioxide'].max()
density_min = data['density'].min()
density_max = data['density'].max()
sulphates_min = data['sulphates'].min()
sulphates_max = data['sulphates'].max()
alcohol_min = data['alcohol'].min()
alcohol_max = data['alcohol'].max()
quality_min = data['quality'].min()
quality_max = data['quality'].max()

print(fixed_min, fixed_max, volatile_min, volatile_max, citric_min, citric_max, residual_min, residual_max, chlorides_min, chlorides_max, free_min, free_max,total_min, total_max,density_min, density_max,sulphates_min, sulphates_max,alcohol_min, alcohol_max,quality_min, quality_max)

"""# Normalization"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'sulphates', 'alcohol', 'quality']
data[num_vars] = scaler.fit_transform(data[num_vars])

data.head()

"""Data Split"""

#split dataset in features and target variable
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','sulphates','alcohol']
X = data[feature_cols] 
y = data.quality

# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

"""Random Forest Model Training"""

#Fitting Decision Tree classifier to the training set  
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 3, criterion="entropy")  
classifier.fit(X_train, y_train)

"""Prediction"""

#Predicting the test set result  
y_pred= classifier.predict(X_test)

"""Model Metrics"""

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
cm

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""Save the Model"""

import pickle
# Save the model
filename = 'random_forest_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

"""Deployment"""

list_of_columns = data.columns
input_data=pd.DataFrame(columns=list_of_columns)
input_data.drop(['quality'], axis='columns', inplace=True)


input_data.at[0, 'fixed acidity'] = float(input('Enter Fixed Acidity Content '))
input_data.at[0, 'volatile acidity'] = float(input('Enter Volatile Acidity Content '))
input_data.at[0, 'citric acid'] = float(input('Enter Citric Acid Content '))
input_data.at[0, 'residual sugar'] = float(input('Enter residual Sugar Content '))
input_data.at[0, 'chlorides'] = float(input('Enter Chlorides Content '))
input_data.at[0, 'free sulfur dioxide'] = float(input('Enter Free Sulphur Dioxide Content '))
input_data.at[0, 'total sulfur dioxide'] = float(input('Enter total Sulphur Dioxide Content'))
input_data.at[0, 'density'] = float(input('Enter Density Content'))
input_data.at[0, 'sulphates'] = float(input('Enter Sulphates Content'))
input_data.at[0, 'alcohol'] = float(input('Enter Alcohol Content'))

"""Denormalization"""

input_data['fixed acidity']=(input_data['fixed acidity']-fixed_min)/(fixed_max-fixed_min)
input_data['volatile acidity']=(input_data['volatile acidity']-volatile_min)/(volatile_max-volatile_min)
input_data['citric acid']=(input_data['citric acid']-citric_min)/(citric_max-citric_min)
input_data['residual sugar']=(input_data['residual sugar']-residual_min)/(residual_max-residual_min)
input_data['chlorides']=(input_data['chlorides']-chlorides_min)/(chlorides_max-chlorides_min)
input_data['free sulfur dioxide']=(input_data['free sulfur dioxide']-free_min)/(free_max-free_min)
input_data['total sulfur dioxide']=(input_data['total sulfur dioxide']-total_min)/(total_max-total_min)
input_data['density']=(input_data['density']-density_min)/(density_max-density_min)
input_data['sulphates']=(input_data['sulphates']-sulphates_min)/(sulphates_max-sulphates_min)
input_data['alcohol']=(input_data['alcohol']-alcohol_min)/(alcohol_max-alcohol_min)

y_pred =  classifier.predict(input_data)
quality = y_pred*(quality_max-quality_min)+quality_min
if quality == 1:
  print('The wine quality is good.')
else:
  print('The quality of wine is bad.')