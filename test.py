import pandas as pd
import numpy as np


#  Loading the training dataset.

df_train = pd.read_csv('data/bank/train.csv')

#  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the training set.

df_train=df_train.replace('unknown',np.nan)
df_train = df_train.fillna(df_train.mode().iloc[0])

print(df_train)

#  Loading the test dataset.

df_test = pd.read_csv('data/bank/test.csv')

#  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the test set.

# df_test=df_test.replace('unknown',np.nan)
# df_test = df_test.fillna(df_test.mode().iloc[0])

# print(df_test)