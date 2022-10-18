import pandas as pd
from id3_functions import id3, predict

df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/train.csv')
df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/DecisionTree/data/car/test.csv')

metrics = ['entropy', 'majority error', 'gini index']
t_ds = [1, 2, 3, 4, 5, 6]

dict_train = {'entropy': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None},
         'majority error': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None},
         'gini index': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}}

dict_test = {'entropy': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None},
         'majority error': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None},
         'gini index': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}}

for metric in metrics:
    for t_d in t_ds:

        # train
        tree= id3(df_train, metric = metric, tree_depth = t_d)
        # print(tree)

        #  Accuracy
        train_accuracy = predict(df_train, tree)
        test_accuracy = predict(df_test, tree)
        # print(train_accuracy, test_accuracy)
        dict_train[metric][t_d] = train_accuracy
        dict_test[metric][t_d] = test_accuracy
        # print(t_d, 'training done!')

# print(dict_test)
print('Accuracy in training set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_train))
print('Accuracy in test set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_test))



























# b = id3(df_train, metric = 'majority error', tree_depth = 1)
# print(b)

# Y_label = []
# for i in range(len(df_train)):
#   inst = df_train.iloc[i,:]
#   prediction = predict_core(inst, tree)
#   Y_label.append(prediction)
# # print(Y_label)

# def vals(x):
#     if isinstance(x, dict):
#         result = []
#         for v in x.values():
#             result.extend(vals(v))
#         return result
#     else:
#         return [x]

# d = vals(tree)
# print(d)