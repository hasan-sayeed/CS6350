import pandas as pd
from id3_functions import id3, predict

df_result_train = pd.DataFrame(columns = ['Information Gain', 'Majority Error', 'Gini Index'],
                                index = ['1', '2', '3', '4', '5', '6'])
df_result_test = pd.DataFrame(columns = ['Information Gain', 'Majority Error', 'Gini Index'],
                                index = ['1', '2', '3', '4', '5', '6'])

df_train = pd.read_csv('data/bank/train.csv')
df_test = pd.read_csv('data/bank/test.csv')

metrics = ['entropy', 'majority error', 'gini index']
t_ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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

        #  Accuracy
        train_accuracy = predict(df_train, tree)
        test_accuracy = predict(df_test, tree)
        print(train_accuracy, test_accuracy)
        dict_train[metric][t_d] = train_accuracy
        dict_test[metric][t_d] = test_accuracy
        print(t_d, 'training done!')

print(dict_train)
print('Accuracy in training set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_train))
print('Accuracy in test set for 3 different heuristics. Index value indicates tree depth.\n', pd.DataFrame.from_dict(dict_test))