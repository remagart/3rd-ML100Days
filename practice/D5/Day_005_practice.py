# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:05:01 2019

@description: practice D5

@author: jfmamjjasond
"""

#%%

5-1

#%%

import pandas as pd

data = {
        'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'visitor': [139, 237, 326, 456]
        }

visitors_1 = pd.DataFrame(data)
print(visitors_1)

visitors_1.groupby(by="weekday")['visitor'].mean()

#%%

import pandas as pd
import numpy as np



data = {
        "country": ["China","Norway","Taiwan","U.S.A"],
        "people": np.random.randint(1000000,1000000000,size=4)
        }

data = pd.DataFrame(data)
print(data)

id = data["people"].idxmax()
print("Most people of country is %s" %(data.loc[id,"country"]))

#%%

5-2

#%%

with open("data/example.txt", "r",encoding="utf-8",errors='ignore') as f:
    data = f.readlines()
print(data)

#%%

data = []
with open("data/example.txt", "r",encoding="utf-8",errors='ignore') as f:
    for line in f:
        line = line.replace("\n","").split(",")
        data.append(line)

print(data)

df = pd.DataFrame(data[1:])
df.columns = data[0]
df

#%%

import requests

target_url = "https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt"

response = requests.get(target_url)
data = response.text

arr = []
arr = data.split("\n")
dataArray = []
for item in arr:
    line = item.split("\t")
    dataArray.append(line)
dataArray

#%%

df = pd.DataFrame(dataArray)
df.columns = ["name","url"]
df.head()

















