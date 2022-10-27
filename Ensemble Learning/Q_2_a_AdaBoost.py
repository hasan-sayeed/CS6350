import pandas as pd
import numpy as np
from id3_functions import id3_e, predict, predict_train, get_bank_data
import id3_functions_mod as dt 
import math 
import matplotlib.pyplot as plt 


#  First weight
def first_w(df):
	w = [1/len(df)]*len(df)
	df['label'] = df['label']*w
	return df, w

def decision_stump(df_train, df_test, actual_label):
	tree= id3_e(df_train, metric = 'entropy', tree_depth = 1)
	train_error, train_pred, loss = predict_train(df_train, tree, actual_label)
	test_error, test_pred = predict(df_test, tree)
	return train_pred, test_pred, loss, train_error, test_error

def category_to_numerical_features(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 

# total error, epsilon
def epsilon(loss, w):
	epsilon = sum([x*y for x,y in zip(loss, w)])
	return epsilon

# column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
# types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
# 		'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

# alpha
def alpha(ep):
	alpha = (np.log((1 - ep)/ep))/2
	return alpha

#  Next weight
def next_w(df, alph, pred, actual_label, w):
	w = w*np.exp([-alph * i for i in [x*y for x,y in zip(pred,actual_label)]])
	w = [x/sum(w) for x in w]
	df['label'] = actual_label
	df['label'] = df['label']*w
	return w, df

# train_data =  pd.read_csv('C:/Users/hasan/OneDrive - University of Utah/Desktop/CS6350-main/Ensemble Learning/Adaboost/bank/train.csv', names=column_names, dtype=types)
# test_data =  pd.read_csv('C:/Users/hasan/OneDrive - University of Utah/Desktop/CS6350-main/Ensemble Learning/Adaboost/bank//test.csv', names=column_names, dtype=types)
# numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
# train_data = category_to_numerical_features(train_data, numerical_features)
# test_data = category_to_numerical_features(test_data, numerical_features)

def adaboost(df_train, df_test, actual_label, w):
	train_pred, test_pred, loss, decision_stump_trn_error, decision_stump_tst_error = decision_stump(df_train, df_test, actual_label)
	ep = epsilon(loss, w)
	alph = alpha(ep)
	w, df_train = next_w(df_train, alph, train_pred, actual_label, w)
	return train_pred, test_pred, loss, ep, alph, w, df_train, decision_stump_trn_error, decision_stump_tst_error

# def adaboost(df_train, df_test, actual_label, w):
#   train_pred, test_pred, loss, decision_stump_trn_error, decision_stump_tst_error = decision_stump(df_train, df_test, actual_label)
#   ep = epsilon(loss, w)
# #   alph = alpha(ep)
# #   w, df_train = next_w(df_train, alph, train_pred, actual_label, w)
#   return train_pred, test_pred, loss, ep, w, df_train, decision_stump_trn_error, decision_stump_tst_error

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

T = 5    #  Number of trees you want!

train_size, test_size = len(train_data),len(test_data)
alphas = [0 for x in range(T)]
weights = np.array([1/train_size for x in range(train_size)])

train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]
train_errorsT, test_errorsT = [0 for x in range(T)], [0 for x in range(T)]

test_p = np.array([0 for x in range(test_size)])

train_p = np.array([0 for x in range(train_size)])
for t in range(T):
	dt_g = dt.Mod_ID3(option=0, max_depth=1)
			
	dt_c = dt_g.construct_dt(train_data, features_dict, label_dict, weights)

	# train errors
	train_data['pred_label']= dt_g.predict(dt_c, train_data)
	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	err = 1 - len(train_data[train_data['result'] == 1]) / train_size
	train_errors[t] = err

	# test errors
	test_data['pred_label']= dt_g.predict(dt_c, test_data)
	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	test_errors[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size
	

	cascade = {}

	# weighted errors and alphas
	tmp = train_data.apply(lambda row: 1 if row['y'] == row['pred_label'] else -1, axis=1)
	tmp = np.array(tmp.tolist())
	w = weights[tmp == -1]
	err = np.sum(w)

	alpha = 0.5 * math.log((1-err)/err)
	alphas[t] = alpha 

	# get new weights 
	weights = np.exp(tmp * -alpha) * weights
	total = np.sum(weights)
	weights = weights/total

	w_weigts = {}

	pred_label = np.array(train_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	train_p = train_p+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[train_p > 0] = 'yes'
	pred_label[train_p <= 0] = 'no'
	train_data['pred_label'] = pd.Series(pred_label)

	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	train_errorsT[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size


	#  test data 
	
	pred_label = np.array(test_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	test_p = test_p+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[test_p > 0] = 'yes'
	pred_label[test_p <= 0] = 'no'
	test_data['pred_label'] = pd.Series(pred_label)

	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	
	test_errorsT[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size

print()
print()
print()
print("________________ Part 2(a) ________________")
print('Test error: ', test_errorsT[-1])


fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(train_errors,  color='r', label='training data')
ax1.set_xlabel('T')
ax1.set_ylabel('error rate')
ax1.plot(test_errors,  color='g', label='test data')

ax1.set_title("Individual tree prediction results")
ax1.legend()

fig2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(train_errorsT,  color='r', label='training data', linewidth = 2)
ax2.set_ylabel('error rate')
ax2.set_xlabel('T')
ax2.plot(test_errorsT,  color='g', label='test data', linewidth = 2)

ax2.set_title("All decision trees prediction results")
ax2.legend()

# plt.draw()
plt.show()