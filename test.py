import pandas as pd
import numpy as np
from id3_functions import id3, predict


#  Loading the training dataset.

df_train = pd.read_csv('data/car/PlayTennis.csv')

# #  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the training set.

# df_train_2=df_train.replace('unknown',np.nan)
# df_train_3 = df_train_2.fillna(df_train_2.mode().iloc[0])

# print(df_train_3)

# #  Loading the test dataset.

# df_test = pd.read_csv('data/bank/test.csv')

#  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the test set.

# df_test=df_test.replace('unknown',np.nan)
# df_test = df_test.fillna(df_test.mode().iloc[0])

# print(df_test)

tree= id3(df_train, metric = 'entropy', tree_depth = 2)
print(tree)