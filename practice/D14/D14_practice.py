#%%
## D14 practice
#%%
import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt


x = np.random.randint(0,50,1000)
y = np.random.randint(0,50,1000)

np.corrcoef(x,y)

#%%
plt.scatter(x,y)


#%%

# 隨機生成 1000 個介於 0~50 的數 x
x = np.random.randint(0, 50, 1000)

# 這次讓 y 與 x 正相關，再增加一些雜訊
y = x + np.random.normal(0, 10, 1000)

print(np.corrcoef(x,y))
plt.scatter(x,y)


