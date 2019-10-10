#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:49:27 2019
@description: D6 practice
@author: richard
"""

#%%

import os
import numpy as np
import pandas as pd


dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

app_train.dtypes.value_counts()

#%%

#app_train.dtypes
#app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)
app_train.select_dtypes(include=["object"])
count = 0
for col in app_train:
    if app_train[col].dtype == "object":
        print(col)
        print(list(app_train[col].unique()))
        print("====\n\n\n====")
        if len(list(app_train[col].unique())) <= 2:
#            print(app_train[col])
            count = count + 1
            
            
print(count)

print("===")

#print(app_train["NAME_EDUCATION_TYPE"])

#%%

from sklearn.preprocessing import LabelEncoder


app_train = pd.read_csv(f_app_train)
le = LabelEncoder()

for col in app_train:
    if app_train[col].dtype == "object":
        if len(list(app_train[col].unique())) <= 2:
#            le.fit(app_train[col])
            app_train[col] = le.fit_transform(app_train[col])
#            app_train[col] = le.transform(app_train[col])
            
print(app_train["CODE_GENDER"].head())


app_train = pd.get_dummies(app_train)

print(app_train['CODE_GENDER_F'])


#%%
## HW

import os
import numpy as np
import pandas as pd

dir_data = "./data"
f_app_train = os.path.join(dir_data,"application_train.csv")
app_train = pd.read_csv(f_app_train)

sub_train = pd.DataFrame(app_train["WEEKDAY_APPR_PROCESS_START"])
#print(sub_train.shape)
#sub_train.head()

sub_train = pd.get_dummies(sub_train)

#print(sub_train["WEEKDAY_APPR_PROCESS_START_MONDAY"])

print(sub_train.shape)
sub_train.head()





