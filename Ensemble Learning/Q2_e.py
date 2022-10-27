import pandas as pd
import matplotlib.pyplot as plt
from id3_functions_ensamble import predict
from Random_forrest_functions import random_forrest, random_forrest_predict, bias, variance
from id3_functions import proccess_train_for_numerical_value, proccess_test_for_numerical_value

df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/bank/train.csv')
df_train, train_median = proccess_train_for_numerical_value(df_train)
df_train['label'] = df_train['label'].map({'yes': 1, 'no': -1})

df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/bank/test.csv')
df_test = proccess_test_for_numerical_value(df_test, train_median)
df_test['label'] = df_test['label'].map({'yes': 1, 'no': -1})

bagged_tr_y_pred = {}
bagged_tst_y_pred = {}
first_trees = {}

for i in range(100):
    tree = random_forrest(df_train, n =4, n_trees=500)
    first_trees[i] = tree[0]
    tr_e, bagged_tr_y_pred[i] = random_forrest_predict(df_train, tree)
    # train_error.append(tr_e)
    tst_e, bagged_tst_y_pred[i] = random_forrest_predict(df_test, tree)
    # test_error.append(tst_e)
# print(len(first_trees))
# print(first_trees)

#  For 100 sigle trees
single_e, single_y_pred = random_forrest_predict(df_test, first_trees)
single_bias = bias(df_test, single_y_pred)
single_var = variance(single_y_pred)
gcd_single = single_bias + single_var

#  For 100 bagged predictor
bagged_tst_y_pred_list = pd.DataFrame(bagged_tst_y_pred).mode(axis=1)[0].tolist()
bagged_bias = bias(df_test, bagged_tst_y_pred_list)
bagged_var = variance(bagged_tst_y_pred_list)
gcd_bagged = bagged_bias + bagged_var

print('Bias for single tree:', single_bias)
print('Varience for single tree:', single_var)
print('General squared error for single tree:', gcd_single)
print('Bias for forrest:', bagged_bias)
print('Varience for forrest:', bagged_var)
print('General squared error for forrest:', gcd_bagged)