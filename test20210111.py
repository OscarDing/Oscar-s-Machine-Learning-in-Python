author = "Oscar Ding"

import numpy as np
import pandas as pd

utdid = np.array(range(5000,6000))

# ab test 1
prime_1 = 487
utdid_1 = (utdid % prime_1)
utdid_1_1 = utdid_1 % 100

np.bincount(utdid_1_1)
np.unique(utdid_1_1)

# df_1 = pd.DataFrame(data={'bucket': np.array(range(0, 100)), 'cnt': np.bincount(utdid_1_1)})
df_1 = pd.DataFrame(data={'bucket': utdid_1_1, 'utdid': utdid}).sort_values(by=['bucket'])
df_1['bucket'].value_counts(ascending=True)
# np.sum(np.where(utdid_1_1 == 92, 1, 0))



prime_2 = 941  # 83 491
utdid_2 = (utdid % prime_2)
utdid_2_1 = utdid_2 % 100

df_2 = pd.DataFrame(data={'bucket': utdid_2_1, 'utdid': utdid}).sort_values(by=['bucket'])
df_2['bucket'].value_counts(ascending=True)