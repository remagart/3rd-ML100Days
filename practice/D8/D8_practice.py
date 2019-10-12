#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:41:48 2019

@description D8 practice
@author: richard
"""

#%%

import os
import numpy as np
import pandas as pd

dir = './data/'
f_app_train = os.path.join(dir,'application_train.csv')

f_app_train = pd.read_csv(f_app_train)
pd.set_option('display.max_columns', 500)
pd.set_option("display.max_rows",500)


import matplotlib.pyplot as plt

#f_app_train.shape
f_app_train.head()

#x = f_app_train.dtypes
#x

test = f_app_train["EXT_SOURCE_2"]

test_m = test.mean()
test_std = test.std()

print(f"mean: {test_m} ; std: {test_std}")

plt.hist(test_m,color="red",label="mean")
plt.hist(test_std,color="green",label="std")

plt.show()








