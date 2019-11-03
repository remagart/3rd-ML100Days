#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = "./data/"
file = os.path.join(dir,"application_train.csv")
app_train = pd.read_csv(file)
app_train.head()


#%%



sub_df = app_train[app_train["DAYS_EMPLOYED"] != 365243]

sub_df["DAYS_EMPLOYED"]

plt.plot(sub_df["DAYS_EMPLOYED"]/(-365), sub_df['AMT_INCOME_TOTAL'], '.')
plt.xlabel('Days of employed (year)')
plt.ylabel('AMT_INCOME_TOTAL (raw)')
plt.show()


#%%

corr = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'])
print(f"Correlation: {corr[0][1]}")


#%%

plt.plot(sub_df["DAYS_EMPLOYED"]/(-365), np.log10(sub_df['AMT_INCOME_TOTAL']), '.')
plt.xlabel('Days of employed (year)')
plt.ylabel('AMT_INCOME_TOTAL (raw)')
plt.show()

corr = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), np.log10(sub_df['AMT_INCOME_TOTAL']))
print(f"Correlation: {corr[0][1]}")

# %%
