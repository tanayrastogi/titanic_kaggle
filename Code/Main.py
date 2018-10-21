#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:16:13 2018

@author: tanay
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np

path = '/home/tanay/Documents/Titanic_Kaggle'



def read_data(path):
    # Reading data from CSV

    print('')
    print('Reading data ...')
    print('------------------------------------')
    
    data = pd.read_csv(path, index_col = 0)
    
    print(data.head())
    print('------------------------------------')
    print('Size of data: ', data.shape)
    print('Columns headers: ')
    print(data.dtypes)
    print('Number of NaN rows: ', len(data[data.isnull().any(axis = 1)]))
    
    return data


def bargraphNaN(data):
    
    # Plotting the bar graph
    plt_data = []
    for i in data.columns:
        plt_data.append(100* (len(data[i][data[i].isnull()]) / len(data)))
        
    fig, ax = plt.subplots()
    ind = np.arange(1, len(data.columns)+1)
    
    # show the figure, but do not block
    plt.bar(ind, plt_data, width=0.75)
    ax.set_xticks(ind)
    ax.set_xticklabels(data.columns)
    ax.set_ylim([0, 100])
    plt.xticks(rotation=90)
    ax.set_title('Missing data Percentage')
    ax.set_ylabel('Percentage Missing')
    for i, v in enumerate(plt_data):
        ax.text(i+1, v, str('{0:.2f}'.format(v)), color='red', fontweight='bold')




# Training data
train = read_data(path+'/Data/train.csv')
# Test data
test = read_data(path+'/Data/test.csv')




types = ['category',    # Survived
         'category',    # Pclas
         'object',      # Name
         'category',    # Sex
         'float',       # Age
         'int',         # SibSp
         'int',         # Parch
         'object',      # Ticket
         'float',       # Fare
         'object',      # Cabin
         'category',    # Embarked
         ]


## Define Dtypes for the columns
# For training set
train_types = types
for i in range(0, len(train.columns)):
    train[train.columns[i]] = train[train.columns[i]].astype(train_types[i])
# For test set
test_types = types[1:]  
for i in range(0, len(test.columns)):
    test[test.columns[i]] = test[test.columns[i]].astype(test_types[i])

  
# Plotting the bar graph
#bargraphNaN(train)
#bargraphNaN(test)


# Bar graph for survivials 
fig, sur_ax = plt.subplots()
sur_ax = train['Survived'].value_counts().plot(kind='bar')
sur_ax.set_xticklabels(['Perished', 'Survied'])
sur_ax.set_title('Survivals: Train data')

# Bar graph for Passenger Class 
fig, pcls_ax = plt.subplots()
pcls_ax = train['Pclass'].value_counts().sort_index().plot(kind='bar')
pcls_ax.set_xticklabels(['First', 'Second','Third'])
pcls_ax.set_title('Passengers in class: Train data')

# Bar graph for Gender 
fig, gnd_ax = plt.subplots()
gnd_ax = train['Sex'].value_counts().sort_index().plot(kind='bar')
gnd_ax.set_title('Gender: Train data')

# Bar graph for Age
fig, age_ax = plt.subplots()
age_ax = train['Age'].hist(bins=8, rwidth=0.9)
age_ax.set_title('Age: Train data')