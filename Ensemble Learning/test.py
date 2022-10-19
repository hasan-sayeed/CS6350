import pandas as pd
import numpy as np
from statistics import mode
from id3_functions import id3, vals, predict, proccess_train_for_numerical_value, proccess_test_for_numerical_value, me_of_total, me_of_attribute


#  Loading the training dataset.

df = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/PlayTennis.csv')
df['label'] = df['label'].map({'Yes': 1, 'No': -1})
actual_label = df['label'].values.tolist()
# print(actual_label)

#  First weight


first_w = [1/len(df)]*len(df)
#  Weighted actual
df['label'] = df['label']*first_w
print(df)

tree= id3(df, metric = 'entropy', tree_depth = 1)
error, pred, loss = predict(df, tree, actual_label)
print(pred)
print(loss)

# total error epsilon
epsilon = sum([x*y for x,y in zip(loss,first_w)])

print(epsilon)

# alpha
alpha = (np.log((1 - epsilon)/epsilon))/2

print(alpha)

#  Next weight

w = first_w*np.exp([-alpha * i for i in [x*y for x,y in zip(pred,actual_label)]])
w = w/sum(w)
# print(w)
# print(sum(w))

































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