import pandas as pd
import numpy as np
from id3_functions import predict, ID3, get_bank_data
import matplotlib.pyplot as plt 

#  bagged tree

def bagging_model(df, n, n_trees, replacement = True):
    
    # n = int, number of examples in each sample
    # replacement = bool, True or False

    trees = {} ## models define
    ## set the seed to repeat the boot strapping
    for i in range(n_trees):
        #sample from X_train, y_train
        df_sampled = df.sample(n, replace = replacement, ignore_index=True)
        trees[i] = ID3(df_sampled, metric = 'entropy', tree_depth = 20)
    return(trees)

# column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
# types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
# 		'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

# # load train data 
# train_data =  pd.read_csv('data/bank/train.csv', names=column_names, dtype=types)
# # load test data 
# test_data =  pd.read_csv('data/bank/test.csv', names=column_names, dtype=types)

# numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# train_data = category_to_numerical_features(train_data, numerical_features)
# test_data = category_to_numerical_features(test_data, numerical_features)




# features_dict = {}
# features_dict['age'] = [0, 1]
# features_dict['balance'] = [0, 1]
# features_dict['day'] = [0, 1]
# features_dict['previous'] = [0, 1]
# features_dict['campaign'] = [0, 1]
# features_dict['pdays'] = [0, 1]
# features_dict['duration'] = [0, 1]
# features_dict['loan'] = ['yes', 'no']
# features_dict['default'] = ['yes', 'no']
# features_dict['housing'] = ['yes', 'no']
# features_dict['job'] = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services']
# features_dict['marital'] = ['married','divorced','single']
# features_dict['education'] = ['unknown', 'secondary', 'primary', 'tertiary']
# features_dict['contact'] = ['unknown', 'telephone', 'cellular']
# features_dict['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
# features_dict['poutcome'] = ['unknown', 'other', 'failure', 'success']

# label_dict = {}
# label_dict['y'] = ['yes', 'no']

features_dict, label_dict, train_data, test_data = get_bank_data()

def bias_e(df, y_pred_array):
    targets = df.to_numpy()
    # y_pred_series = pd.Series(y_pred_list)
    # bias = (sum((df[df.columns[-1]].subtract(y_pred_series, axis = 0))**2)) / len(df[df.columns[-1]])
    bias = (sum((targets[:,-1]-y_pred_array)**2))/y_pred_array.size
    return(bias)

T = 500  # number of trees you want!

train_size, test_size = len(train_data),len(test_data)
train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]

test_py = np.array([0 for x in range(test_size)])
train_py = np.array([0 for x in range(train_size)])

def bagging_predict(df, trees):
    targets = df.to_numpy()
    y_pred = {}
    for k in trees.keys():
        _, y_pred[k] = predict(df, trees[k])
    y_pred_array = pd.DataFrame(y_pred).mode(axis=1)[0].to_numpy()

    # error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_pred_list) if x == y) / len(df[df.columns[-1]])

    error = sum(1 - np.isin(targets[:,-1], y_pred_array))/y_pred_array.size
    return(error, y_pred_array)

for t in range(T):

	# sample train data
	part_train = train_data.sample(frac=0.5, replace=True, random_state = t)

	dt_g = ID3(option=0, max_depth=15)
			
	dt_c = dt_g.construct_dt(part_train, features_dict, label_dict)

	# predict train data
	pred_label = dt_g.predict(dt_c, train_data)
	pred_label = np.array(pred_label.tolist())

	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	train_py = train_py+pred_label
	pred_label = pred_label.astype(str)
	pred_label[train_py > 0] = 'yes'
	pred_label[train_py <= 0] = 'no'
	train_data['pred_label'] = pd.Series(pred_label)

	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	
	train_errors[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size

	# predict test data 
	pred_label = dt_g.predict(dt_c, test_data)
	pred_label = np.array(pred_label.tolist())

	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	test_py = test_py+pred_label
	pred_label = pred_label.astype(str)
	pred_label[test_py > 0] = 'yes'
	pred_label[test_py <= 0] = 'no'
	test_data['pred_label'] = pd.Series(pred_label)

	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	
	test_errors[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size

def variance_e(y_pred_array):
    # y_pred_array = np.array(y_pred_list)
    var = np.var(y_pred_array, ddof=1)
    return(var)


print()
print()
print()
print("________________ Part 2(b) ________________")
print('Test error: ', test_errors[-1])
plt.plot(train_errors, label = "training error", c = 'b', linewidth = 2)
plt.plot(test_errors, label = "test error", c = 'r', linewidth = 2)
plt.xlabel('T')
plt.ylabel('error rates')
plt.title('Bagged Tree')
plt.legend()
# plt.show()


# bias variance decomposition for bagged tree

train_size, test_size = len(train_data),len(test_data)
test_p = np.array([[0 for x in range(test_size)] for y in range(100)])
test_p_first = np.array([0 for x in range(test_size)])
for i in range(100):
	train_subset = train_data.sample(n=1000, replace=False, random_state=i)
	for t in range(500):
		# get sample train data 
		sampled = train_subset.sample(frac=1, replace=True, random_state=t)

		dt_generator = ID3(option=0, max_depth=15)
			
		dt_construction = dt_generator.construct_dt(sampled, features_dict, label_dict)

		pred_label = dt_generator.predict(dt_construction, test_data)
		pred_label = np.array(pred_label.tolist())

		pred_label[pred_label == 'yes'] = 1 
		pred_label[pred_label == 'no'] = -1
		pred_label = pred_label.astype(int)
		test_p[i] = test_p[i]+pred_label
		if t==0:
			test_p_first = test_p_first + pred_label


true_y = np.array(test_data['y'].tolist())
true_y[true_y == 'yes'] = 1 
true_y[true_y == 'no'] = -1 
true_y = true_y.astype(int)

# choose first tree 
test_p_first = test_p_first/100 


bias = np.mean(np.square(test_p_first - true_y))
variance = np.sum(np.square(test_p_first - np.mean(test_p_first))) / (test_size -1)
squareError = bias + variance 
print()
print()
print()
print("________________ Part 2(c) ________________")
print("results for 100 single tree predictor-  bias: ", bias, "    variance: ", variance, "    Gen Squared Error: ", squareError)


test_p = np.sum(test_p,axis=0) / 50000


bias = np.mean(np.square(test_p - true_y))
variance = np.sum(np.square(test_p - np.mean(test_p))) / (test_size -1)
squareError = bias + variance 

print()
print()
print("results for 100 bagged tree predictor-   bias: ", bias, "    variance: ", variance, "    Gen Squared Error: " ,squareError)


plt.show()