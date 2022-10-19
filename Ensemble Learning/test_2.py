import pandas as pd
from AdaBoost_functions import first_w, adaboost


#  Loading the training dataset.
df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/PlayTennis.csv')
df_train['label'] = df_train['label'].map({'Yes': 1, 'No': -1})
actual_label = df_train['label'].values.tolist()

df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/PlayTennis.csv')
df_test['label'] = df_test['label'].map({'Yes': 1, 'No': -1})



# initialige first weight and modify dataframe
df_train, w = first_w(df_train)
result = []
for i in range(3):
    pred, loss, ep, alph, w, df_train, decision_stump_trn_error, decision_stump_tst_error = adaboost(df_train, df_test, actual_label, w)
    result.append([alph*j for j in pred])
# pred, loss, ep, alph, w, df = adaboost(df, actual_label, w)
print(pred, loss, ep, alph, w, decision_stump_trn_error, decision_stump_tst_error, [alph*j for j in pred])
print(df_train)
print(result)
print([sum(i) for i in zip(*result)])

































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