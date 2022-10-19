import pandas as pd
import numpy as np
from id3_functions import proccess_train_for_numerical_value, proccess_test_for_numerical_value
from AdaBoost_functions import first_w, adaboost


#  Loading the training dataset.
df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/bank/train.csv')
df_train, train_median = proccess_train_for_numerical_value(df_train)
df_train['label'] = df_train['label'].map({'yes': 1, 'no': -1})
actual_label = df_train['label'].values.tolist()

df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/bank/test.csv')
df_test = proccess_test_for_numerical_value(df_test, train_median)
df_test['label'] = df_test['label'].map({'yes': 1, 'no': -1})
# print(df_test)



# initialige first weight and modify dataframe
df_train, w = first_w(df_train)
result_train = []
result_test = []
loss = []
for i in range(50):
    train_pred, test_pred, loss, ep, alph, w, df_train, decision_stump_trn_error, decision_stump_tst_error = adaboost(df_train, df_test, actual_label, w)
    # train_pred, test_pred, loss, ep, w, df_train, decision_stump_trn_error, decision_stump_tst_error = adaboost(df_train, df_test, actual_label, w)
    result_train.append([alph*j for j in train_pred])
    result_test.append([alph*j for j in test_pred])
    # result_train.append([5*j for j in train_pred])
    # result_test.append([5*j for j in test_pred])
    # loss.append(loss)
final_pred_train = np.sign([sum(i) for i in zip(*result_train)])
final_pred_test = np.sign([sum(i) for i in zip(*result_test)])

# print(result_train)
# # print(result_test)
# print(final_pred_train)
# print(final_pred_test)
# print(loss)

error_train = 1 - sum(1 for x,y in zip(actual_label,final_pred_train) if x == y) / len(actual_label)
error_test = 1 - sum(1 for x,y in zip(df_test[df_test.columns[-1]],final_pred_test) if x == y) / len(df_test[df_test.columns[-1]])
print(error_train)
print(error_test)

































# df_m = df
# print(df)
# df_m['weight'] = 1/len(df_m)
# df_m['weight*actual'] = df_m['weight']*df_m['label']
# # column_to_move = df.pop("label")
# # df.insert(6, "label", column_to_move)
# df_m_2 = df
# print(df)
# df_m_2['label'] = df_m['weight*actual']
# print(df)
# # print(df)
# # print(df_m)
# # print(df_m_2)