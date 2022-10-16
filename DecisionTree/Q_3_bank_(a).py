import pandas as pd
import numpy as np
from id3_functions import id3, predict, proccess_train_for_numerical_value, proccess_test_for_numerical_value

#  Loading the training dataset.

df_train = pd.read_csv('data/bank/train.csv')
df_train_2, train_median = proccess_train_for_numerical_value(df_train)

#  Loading the test dataset.

df_test = pd.read_csv('data/bank/test.csv')
df_test_2 = proccess_test_for_numerical_value(df_test, train_median)

metrics = ['entropy', 'majority error', 'gini index']
t_ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

dict_train = {'entropy': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None},
         'majority error': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None},
         'gini index': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None}}

dict_test = {'entropy': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None},
         'majority error': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None},
         'gini index': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None}}

for metric in metrics:
    for t_d in t_ds:

        # train
        tree= id3(df_train_2, metric = metric, tree_depth = t_d)

        #  Accuracy
        train_accuracy = predict(df_train_2, tree)
        test_accuracy = predict(df_test_2, tree)
        print(train_accuracy, test_accuracy)
        dict_train[metric][t_d] = train_accuracy
        dict_test[metric][t_d] = test_accuracy
        print(t_d, 'training done!')

# print(dict_train)
print('Accuracy in training set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_train))
print('Accuracy in test set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_test))