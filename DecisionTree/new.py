import pandas as pd
import numpy as np
from id3_functions import id3, predict, proccess_train_for_numerical_value, proccess_test_for_numerical_value

df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/bank/train.csv')
df_train_2, train_median = proccess_train_for_numerical_value(df_train)

print(df_train)
print(df_train_2)