import pandas as pd
import matplotlib.pyplot as plt
from id3_functions_ensamble import predict
from Bagging_functions import bagging_model, bagging_predict, bias, variance
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
    tree = bagging_model(df_train, n =1000, n_trees=500, replacement = False)
    first_trees[i] = tree[0]
    tr_e, bagged_tr_y_pred[i] = bagging_predict(df_train, tree)
    # train_error.append(tr_e)
    tst_e, bagged_tst_y_pred[i] = bagging_predict(df_test, tree)
    # test_error.append(tst_e)
# print(len(first_trees))
# print(first_trees)

#  For 100 sigle trees
single_e, single_y_pred = bagging_predict(df_test, first_trees)
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
print('Bias for bagged tree:', bagged_bias)
print('Varience for bagged tree:', bagged_var)
print('General squared error for bagged tree:', gcd_bagged)

# fig = plt.figure(1, figsize=(5,5))
# plt.plot(n_trees, train_error, marker = 'o', label = 'train error', mfc = 'white', linestyle= '--', alpha = 0.7)
# plt.plot(n_trees, test_error, marker = 'o', label = 'test error' , mfc = 'white', linestyle= '--', alpha = 0.7)

# plt.legend()
# plt.show()







# Code Graveyeard


# def bagging_model(X_train, y_train, size_sample, n_trees=3):
#     m = {} ## models define
#     ## set the seed to repeat the boot strapping
#     np.random.seed(20)
#     index = np.arange(X_train.shape[0])
#     for i in range(n_trees):
#         #sample from X_train, y_train
#         index_sampled = np.random.choice(index, size=size_sample, replace=True)
#         X_train_sample = X_train.loc[index_sampled,:]
#         y_train_sample = y_train.loc[index_sampled]
#         m[i] = decision_tree_model(X_train_sample, y_train_sample,model_name='IG',max_depth =16)
#     return(m)
# def bagging_predict(X,model):
#     y_pred = {}
#     for k in model.keys():
#         y_pred[k] = predict(X,model[k])
#     return(np.max(pd.DataFrame(y_pred),axis=1))


# print(df)
# index = np.arange(df.shape[0])
# # print(index)
# trees = {}
# for i in range(3):
#     # index_sampled = np.random.choice(index, size=17, replace=True)
#     # print(index_sampled)
#     df_sampled = df.sample(frac = 1, replace = True, ignore_index=True)
#     df_sampled
#     # print(df_sampled)
#     trees[i] = id3(df_sampled, metric = 'entropy', tree_depth = 14)
# print(trees)

# def bagging_model(df, frac_sample=1, n_trees=3):
#     trees = {} ## models define
#     ## set the seed to repeat the boot strapping
#     for i in range(n_trees):
#         #sample from X_train, y_train
#         df_sampled = df.sample(frac = frac_sample, replace = True, ignore_index=True)
#         trees[i] = id3(df_sampled, metric = 'entropy', tree_depth = 14)
#     return(trees)

# def bagging_predict(df, trees):
#     y_pred = {}
#     for k in trees.keys():
#         _, y_pred[k] = predict(df, trees[k])
#     y_p = pd.DataFrame(y_pred)
#     return(y_p.mode(axis=1)[0].tolist())