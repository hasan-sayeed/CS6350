import pandas as pd
import numpy as np
from id3_functions_rf import id3_e, predict, ID3, get_bank_data
import matplotlib.pyplot as plt 


def random_forrest(df, n, n_trees):
	trees = {}
	integrated_t = []
	for i in range(n_trees):
		
		df1 = df.iloc[:, :-1]
		df2 = df.iloc[:, -1:]
		df3 = df.iloc[:, :-1]
		df_sampled = df1.sample(n, replace = False, axis = 1)
		df_n = pd.concat([df_sampled, df2], axis=1)
		df_c = df3.iloc[:, :-1]
		integrated_t.append(1)
		trees[i] = id3_e(df_n, metric = 'entropy', tree_depth = 20)
	return(trees)

def random_forrest_predict(df, trees):
	y_pred = {}
	for k in trees.keys():
		train_error, y_pred[k] = predict(df, trees[k])
	# print(pd.DataFrame(y_pred))
	y_pred_list = pd.DataFrame(y_pred).mode(axis=1)[0].tolist()
	error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_pred_list) if x == y) / len(df[df.columns[-1]])
	return(error, y_pred_list)

def rf(features_dict, label_dict, train_data, test_data, num_subset, label_name):
	T = 500
	train_size, test_size = len(train_data),len(test_data)
	train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]

	test_py = np.array([0 for x in range(test_size)])
	train_py = np.array([0 for x in range(train_size)])

	for t in range(T):
		sampled = train_data.sample(frac=0.5, replace=True, random_state=t)
		# build tree 
		dt_generator = ID3(option=0, max_depth=15, subset = num_subset)
			
		dt_construction = dt_generator.construct_dt(sampled, features_dict, label_dict)

		# train
		pred_label = dt_generator.predict(dt_construction, train_data)
		pred_label = np.array(pred_label.tolist())

		pred_label[pred_label == 'yes'] = 1 
		pred_label[pred_label == 'no'] = -1
		pred_label = pred_label.astype(int)
		train_py = train_py+pred_label
		pred_label = pred_label.astype(str)
		pred_label[train_py > 0] = 'yes'
		pred_label[train_py <= 0] = 'no'
		train_data['pred_label'] = pd.Series(pred_label)

		train_data['result'] = (train_data[[label_name]].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	
		train_errors[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size
		
		# predict test data 
		pred_label = dt_generator.predict(dt_construction, test_data)
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

	return train_errors, test_errors

# def save_fig(train_errors, test_errors, num_subset, fig_name):
# 	plt.plot(train_errors, label = "training error", c = 'b', linewidth = 2)
# 	plt.plot(test_errors, label = "test error", c = 'r', linewidth = 2)
# 	plt.xlabel('T')
# 	plt.ylabel('error rates')
# 	plt.title('Random Forest '+str(num_subset)+' subsets')
# 	plt.legend()
# 	plt.show()
# 	plt.savefig(fig_name, dpi=300, bbox_inches='tight')

# num_subset = [2 , 4, 6]

# for num in num_subset:
# 	features_dict, label_dict, train_data, test_data = get_bank_data()
# 	train_errors, test_errors = rf(features_dict, label_dict, train_data, test_data, num, 'y')
# 	print()
# 	print()
# 	print('Test error for subset size {s} is: {e}'.format(s = num, e = test_errors[-1]))
# 	# save_fig(train_errors, test_errors, num, 'bank_'+str(num)+'.png')
# 	plt.plot(train_errors, label = "training error", c = 'b', linewidth = 2)
# 	plt.plot(test_errors, label = "test error", c = 'r', linewidth = 2)
# 	plt.xlabel('T')
# 	plt.ylabel('error rates')
# 	plt.title('Random Forest '+str(num_subset)+' subsets')
# 	plt.legend()
# 	plt.show()
	# plt.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.show()

def bias_e(df, y_pred_list):
	y_pred_series = pd.Series(y_pred_list)
	bias = (sum((df[df.columns[-1]].subtract(y_pred_series, axis = 0))**2)) / len(df[df.columns[-1]])
	return(bias)

def variance_e(y_pred_list):
	y_pred_array = np.array(y_pred_list)
	var = np.var(y_pred_array, ddof=1)
	return(var)


# Bias variance decomposition
print()
print()
print("________________ Part 2(e) ________________")
print()

def rf_b_v(features_dict, label_dict, train_data, test_data, num_subset, label_name):
	
	train_size, test_size = len(train_data),len(test_data)
	test_py = np.array([[0 for x in range(test_size)] for y in range(100)])
	test_py_first = np.array([0 for x in range(test_size)])

	for i in range(100):
		train_subset = train_data.sample(n=1000, replace=False, random_state=i)
		for t in range(500):
			sampled = train_subset.sample(frac=0.1, replace=True, random_state=t)

			# build tree 
			dt_generator = ID3(option=0, max_depth=15, subset = num_subset)
			
			dt_construction = dt_generator.construct_dt(sampled, features_dict, label_dict)

			# predict test 
			pred_label = dt_generator.predict(dt_construction, test_data)
			pred_label = np.array(pred_label.tolist())

			pred_label[pred_label == 'yes'] = 1 
			pred_label[pred_label == 'no'] = -1
			pred_label = pred_label.astype(int)
			test_py[i] = test_py[i]+pred_label
			if t==0:
				test_py_first = test_py_first+pred_label


	true_y = np.array(test_data[label_name].tolist())
	true_y[true_y=='yes'] = 1
	true_y[true_y=='no'] = -1
	true_y=true_y.astype(int)

	# predicts first tree 
	test_py_first = test_py_first/100 

	# bias 
	bias = np.mean(np.square(test_py_first - true_y))

	#variance 
	mean = np.mean(test_py_first)
	variance = np.sum(np.square(test_py_first - mean)) / (test_size - 1)
	squaredError = bias+variance

	
	print()	
	print("Decomposition results when subset =", num_subset)
	print()
	print("100 single tree predictor-  bias:  ", bias, "    variance:  ", variance, "    Gen Squared Error:  ", squaredError)

	# random forest 
	test_py = np.sum(test_py,axis=0) / 50000

	# bias 
	bias = np.mean(np.square(test_py - true_y))

	# variance 
	mean = np.mean(test_py)
	variance = np.sum(np.square(test_py - mean)) / (test_size - 1)
	squaredError = bias+variance 
	# print()
	# print()
	print("100 random forest predictor-  bias:  ", bias, "    variance:  ", variance, "    Gen Squared Error:  ", squaredError)

num_subset = [2 , 4, 6]
for num in num_subset:
	features_dict, label_dict, train_data, test_data = get_bank_data()
	rf_b_v(features_dict, label_dict, train_data, test_data, num, 'y')

print()
print()
print("________________ Part 2(d) ________________")
print()


num_subset = [2 , 4, 6]

for num in num_subset:
	features_dict, label_dict, train_data, test_data = get_bank_data()
	train_errors, test_errors = rf(features_dict, label_dict, train_data, test_data, num, 'y')
	
	print()
	print('Test error for subset size {s} is: {e}'.format(s = num, e = test_errors[-1]))
	# save_fig(train_errors, test_errors, num, 'bank_'+str(num)+'.png')
	plt.plot(train_errors, label = "training error", c = 'b', linewidth = 2)
	plt.plot(test_errors, label = "test error", c = 'r', linewidth = 2)
	plt.xlabel('T')
	plt.ylabel('error rates')
	plt.title('Random Forest '+str(num)+' subsets')
	plt.legend()
	plt.show()