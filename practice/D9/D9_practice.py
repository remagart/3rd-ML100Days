#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 01:35:29 2019

@description D9 practice
@author: richard
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = "./data/"
file = os.path.join(dir,"application_train.csv")

print(f"Path of csv file: {file}")

app_train = pd.read_csv(file)
pd.set_option("display.max_columns",500)

app_train["DAYS_EMPLOYED"]

#%%

#(app_train["DAYS_BIRTH"]/(-365)).describe()

(app_train['DAYS_EMPLOYED'] / 365).describe()

#%%
plt.hist(app_train['DAYS_EMPLOYED'])
plt.show()
app_train['DAYS_EMPLOYED'].value_counts()

#%%
sum(app_train["DAYS_EMPLOYED"] == 365243) / len(app_train)

#%%

app_train["DAYS_EMPLOYED_ANOM"] = app_train["DAYS_EMPLOYED"] == 365243

app_train["DAYS_EMPLOYED_ANOM"].value_counts()

#%%

app_train["DAYS_EMPLOYED"].replace({365243: np.nan},inplace = True)


app_train["DAYS_EMPLOYED"]

#%%
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram',color="red");
plt.xlabel('Days Employment');


#%%

x = app_train[~app_train.OWN_CAR_AGE.isnull()]["OWN_CAR_AGE"]

print(x)


plt.hist(x)
plt.show()

x.value_counts()

#%%
app_train[app_train["OWN_CAR_AGE"]>50]["OWN_CAR_AGE"].value_counts()


#%%
# HW
#%%

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir = "./data"

file = os.path.join(dir,"application_train.csv")
app_train = pd.read_csv(file)


dtypes = ["int64","float64"]

for dtype_select in dtypes:
    # 取出符合格式的
    isInclude = list(dtype_select == app_train.dtypes)
    numeric_columns = list( app_train.columns[isInclude])
    
    print(len(numeric_columns))


    # 找出不是0和1的    
    lam = (lambda x:len(x.unique())!=2)
    col = list(app_train[numeric_columns].apply(lam))
    
    numeric_columns = list(app_train[numeric_columns].columns[col])
    
    print("Numbers of remain columns",  len(numeric_columns))
    
    arrange = ['AMT_INCOME_TOTAL',
               'REGION_POPULATION_RELATIVE',
               'OBS_60_CNT_SOCIAL_CIRCLE']
    
    for col in numeric_columns:
        if col  not in arrange : continue
    
        app_train[col].plot.hist(title = col+" - "+dtype_select, bins=20)
        plt.show()


#%%
print(app_train['AMT_INCOME_TOTAL'].describe())

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1,len(x)+1) / len(x)
    return x,y


x,y = ecdf(app_train['AMT_INCOME_TOTAL'])

plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([min(x), max(x) * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()


# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
plt.plot(np.log(x), y)
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

#%%

print(app_train['REGION_POPULATION_RELATIVE'].describe())

x,y = ecdf(app_train['REGION_POPULATION_RELATIVE'])
plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('ECDF')

plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()


#%%

print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

x,y = ecdf(app_train['OBS_60_CNT_SOCIAL_CIRCLE'])

plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([min(x) * 0.95, max(x) * 1.05])
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()

print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))

#%%

app_train["OBS_60_CNT_SOCIAL_CIRCLE"].value_counts()

data = app_train[app_train['OBS_60_CNT_SOCIAL_CIRCLE']<20]['OBS_60_CNT_SOCIAL_CIRCLE']


x,y = ecdf(data)

plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('ECDF')

plt.show()

data.hist()
plt.show()




