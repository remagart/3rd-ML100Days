#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 生成範例用的資料 ()
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                    index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                    index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])

result = pd.concat([df1,df2,df3])
result

#%%

result = pd.concat([df1,df4],axis=1)
result

#%%

result = pd.concat([df1, df4], axis = 1, join = 'inner') # 硬串接
print(result)

#%%
result = pd.merge(df1, df4, how='inner')
print(f"\n\n{df1}\n\n")
print(f"{df4}\n\n")
print(result)
#%%

print(df1)

df1.melt()

#%%

dir = "./data/"
file = os.path.join(dir,"application_train.csv")
print(file)

app_train = pd.read_csv(file)
app_train.head()


#%%

# 取 TARGET 為 1 的
sub_df = app_train[app_train["TARGET"] == 1]
sub_df

#%%

sub_df = app_train.loc[app_train['AMT_INCOME_TOTAL'] > app_train['AMT_INCOME_TOTAL'].mean(), ['SK_ID_CURR', 'TARGET']]
sub_df.head()

#%%
app_train.groupby(['NAME_CONTRACT_TYPE']).size()
# %%

app_train.groupby(['NAME_CONTRACT_TYPE'])['AMT_INCOME_TOTAL'].describe()


#%%
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].mean()

#%%
# 取前 10000 筆作範例: 分別將 AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY 
# 除以根據 NAME_CONTRACT_TYPE 分組後的平均數，
filter = [ "NAME_CONTRACT_TYPE",'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
gb = ['NAME_CONTRACT_TYPE']
app_train.loc[0:10000, filter].groupby(gb).apply(lambda x: x / x.mean())
# app_train.loc[0:10000, filter].groupby(gb)
# %%
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].hist()
plt.show()

# %%


# %%


# %%
