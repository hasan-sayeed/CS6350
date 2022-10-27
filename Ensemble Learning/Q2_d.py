import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
from Random_forrest_functions import random_forrest, random_forrest_predict
from id3_functions import proccess_train_for_numerical_value, proccess_test_for_numerical_value

df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/bank/train.csv')
df_train, train_median = proccess_train_for_numerical_value(df_train)
df_train['label'] = df_train['label'].map({'yes': 1, 'no': -1})

df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/bank/test.csv')
df_test = proccess_test_for_numerical_value(df_test, train_median)
df_test['label'] = df_test['label'].map({'yes': 1, 'no': -1})

n_trees = list(range(1,501))
n_sub_features = [2, 4, 6]
train_error = [[] for x in range(len(n_sub_features))]
test_error = [[] for x in range(len(n_sub_features))]


# def rf_trees(n_tree):
for n_tree in n_trees:
    a=0
    for n in n_sub_features:
        tree = random_forrest(df_train, n = n, n_trees=n_tree)
        tr_e, _ = random_forrest_predict(df_train, tree)
        train_error[a].append(tr_e)
        tst_e, _ = random_forrest_predict(df_test, tree)
        test_error[a].append(tst_e)
        a += 1

# with concurrent.futures.ProcessPoolExecutor() as excecutor:
#     excecutor.map(rf_trees, n_trees)

print(n_trees)
print(train_error[0])
print(train_error[1])
print(train_error[2])

fig = plt.figure(1, figsize=(5,5))
plt.plot(n_trees, train_error[0], label = 'train error', mfc = 'white', linestyle= '--', alpha = 0.7)
plt.plot(n_trees, test_error[0], label = 'test error' , mfc = 'white', linestyle= '--', alpha = 0.7)
plt.xlabel('Error') 
plt.ylabel('number of random tree') 
plt.title("Error vs number of random tree (error subset size = 2)")

fig = plt.figure(2, figsize=(5,5))
plt.plot(n_trees, train_error[1], label = 'train error', mfc = 'white', linestyle= '--', alpha = 0.7)
plt.plot(n_trees, test_error[1], label = 'test error' , mfc = 'white', linestyle= '--', alpha = 0.7)
plt.xlabel('Error') 
plt.ylabel('number of random tree') 
plt.title("Error vs number of random tree (error subset size = 4)")

fig = plt.figure(3, figsize=(5,5))
plt.plot(n_trees, train_error[2], label = 'train error', mfc = 'white', linestyle= '--', alpha = 0.7)
plt.plot(n_trees, test_error[2], label = 'test error' , mfc = 'white', linestyle= '--', alpha = 0.7)
plt.xlabel('Error') 
plt.ylabel('number of random tree') 
plt.title("Error vs number of random tree (error subset size = 6)")

plt.legend()
plt.show()







