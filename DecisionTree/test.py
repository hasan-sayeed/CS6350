import pandas as pd
import numpy as np
from statistics import mode
from id3_functions import id3, vals, predict, proccess_train_for_numerical_value, proccess_test_for_numerical_value, me_of_total, me_of_attribute


#  Loading the training dataset.

df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/PlayTennis.csv')

# #  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the training set.

# df_train_2=df_train.replace('unknown',np.nan)
# df_train_3 = df_train_2.fillna(df_train_2.mode().iloc[0])

# print(df_train_3)

# #  Loading the test dataset.

# df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/bank/test.csv')

#  Replacing 'unknown' value in each column with the majority of other values of the same attribute in the test set.

# df_test=df_test.replace('unknown',np.nan)
# df_test = df_test.fillna(df_test.mode().iloc[0])

# print(df_test)

tree= id3(df_train, metric = 'majority error', tree_depth = 1)
print(tree)


# prediction = mode(vals(tree))
# print(prediction)

# a, b = proccess_train_for_numerical_value(df_train)
# c = proccess_test_for_numerical_value(df_test, train_m=b)
# print (a)
# print(b)

# tree= id3(a, metric = 'entropy', tree_depth = 3)
# print(tree)

# c = me_of_total(df_train)
# print(c)

# d = me_of_attribute(df_train, 'Wind')
# print(d)



# label = df_train.keys()[-1]

# label_vals, label_counts = np.unique(df_train[label], return_counts = True)
# print(label_counts)
# counts = df_train[label].unique()
# print(counts)