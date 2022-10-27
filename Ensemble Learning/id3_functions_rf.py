#  Sources used: lecture slides, github, stackoverflow, towards data science articles 


import math
import copy
import numpy as np
import pandas as pd
from statistics import mode

# Entropy of the whole dataset

def entropy_of_total(df):
	label = df.keys()[-1]
	counts = df[label].unique()    # unique labels
	total_entropy = 0
	for count in counts:
		probability = df[label].value_counts()[count]/len(df[label])
		total_entropy += -probability * np.log2(probability)
	return np.float64(total_entropy)

# Entropy of a single attributes

def entropy_of_attribute(df, attribute):
	label = df.keys()[-1]
	attr_vals = df[attribute].unique()
	target_vals = df[label].unique()
	entropy = 0
	for attr_val in attr_vals:
		entropy_tmp = 0
		for target_val in target_vals:
			num = len(df[attribute][df[attribute] == attr_val][df[label] == target_val])
			den = len(df[attribute][df[attribute] == attr_val])
			probability = num/den
			entropy_tmp += -probability * np.log2(probability + 0.000001)
		entropy += (den/len(df))*entropy_tmp
	return np.float64(entropy)

# Majority error of the whole dataset

def me_of_total(df):
	label = df.keys()[-1]
	label_vals, label_counts = np.unique(df[label], return_counts = True)   # unique labels
	majority_value_count = np.amax(label_counts)
	probability = majority_value_count/len(df[label])
	total_me = 1 - probability
	return np.float64(total_me)

# Majority error of a single attributes

def me_of_attribute(df, attribute):
	label = df.keys()[-1]
	attr_vals = df[attribute].unique()
	me = 0
	for attr_val in attr_vals:
		attr_label_vals, attr_label_counts = np.unique(df[label][df[attribute] == attr_val], return_counts = True)
		majority_value_count = np.amax(attr_label_counts)
		den = len(df[attribute][df[attribute] == attr_val])
		probability = majority_value_count/den
		me_tmp = 1 - probability
		me += (den/len(df))*me_tmp
	return np.float64(me)

# Gini index of the whole dataset

def gi_of_total(df):
	label = df.keys()[-1]
	counts = df[label].unique()    # unique labels
	total_gi = 1
	for count in counts:
		probability = df[label].value_counts()[count]/len(df[label])
		total_gi += -probability**2
	return np.float64(total_gi)

# Gini index of a single attributes

def gi_of_attribute(df, attribute):
	label = df.keys()[-1]
	attr_vals = df[attribute].unique()
	target_vals = df[label].unique()
	gi = 0
	for attr_val in attr_vals:
		gi_tmp = 1
		for target_val in target_vals:
			num = len(df[attribute][df[attribute] == attr_val][df[label] == target_val])
			den = len(df[attribute][df[attribute] == attr_val])
			probability = num/den
			gi_tmp += -probability**2
		gi += (den/len(df))*gi_tmp
	return np.float64(gi)

# Information gain calculation and finding the best attribute to split on

def best_attribute(df, metric):
	IG = []
	if metric=='entropy':
		for key in df.keys()[:-1]:
			IG.append(entropy_of_total(df) - entropy_of_attribute(df, key))
		return df.keys()[:-1][np.argmax(IG)]
	
	elif metric=='majority error':
		for key in df.keys()[:-1]:
			IG.append(me_of_total(df) - me_of_attribute(df, key))
		return df.keys()[:-1][np.argmax(IG)]

	elif metric=='gini index':
		for key in df.keys()[:-1]:
			IG.append(gi_of_total(df) - gi_of_attribute(df, key))
		return df.keys()[:-1][np.argmax(IG)]

# Updated dataframe after every split

def updated_dataframe(df, attribute, val):
	return df[df[attribute] == val].reset_index(drop = True)


# from collections import deque

# def depth(d):
#     queue = deque([(id(d), d, 1)])
#     memo = set()
#     while queue:
#         id_, o, level = queue.popleft()
#         if id_ in memo:
#             continue
#         memo.add(id_)
#         if isinstance(o, dict):
#             queue += ((id(v), v, level + 1) for v in o.values())
#     return level

# constructing the tree. Taking the metric and tree depth from user.

def id3_e(df, metric, tree_depth, tree = None, t_d = 1):
	
	'''
	Parameters
				----------
				metric : str.
						Which varient of information gain you want to use. Possible values are
						"entropy", "majority error" and "gini index".
				tree_depth : str.
						Maximum tree depth you want.
	'''

	node = best_attribute(df, metric)               # Get the best attribute first
	att_counts = np.unique(df[node])
	label = df.keys()[-1]
	# t_d = 1
	if tree is None:
		tree = {}
		tree[node] = {}
	# print(depth(tree))
	for att_count in att_counts:           # Goes to each branch
		updated_data = updated_dataframe(df,node,att_count)
		label_vals, label_counts = np.unique(updated_data[label], return_counts = True)
		# t_d = 1
		# print(att_count, label_vals, label_counts)
		if len(label_counts) == 1:
			tree[node][att_count] = label_vals[0]
		# else:
			# majority_label = np.where(label_counts == np.amax(label_counts))
			# tree[node][att_count] = label_vals[majority_label[0][0]]
		elif t_d == tree_depth:
			majority_label = np.where(label_counts == np.amax(label_counts))
			tree[node][att_count] = label_vals[majority_label[0][0]]
			# t_d = 1
		elif t_d < tree_depth:
			# t_d += 1
			tree[node][att_count] = id3(updated_data, metric, tree_depth, t_d = t_d+1)
			
	return tree


class TNode:
	def __init__(self):
		self.feature = None
		self.children = None
		self.depth = -1
		self.is_leaf_node = False
		self.label = None
	
	# functions to set values
	def set_feature(self, feature):
		self.feature = feature

	def set_children(self, children):
		self.children = children

	def set_depth(self, depth):
		self.depth = depth

	def set_bs(self, feature):
		self.bs = feature

	def set_leaf(self, status):
		self.is_leaf_node = status

	def set_label(self, label):
		self.label = label

	def set_koo(self, feature):
		self.bs = feature

	# functions to return values
	def is_leaf(self):
		return self.is_leaf_node

	def get_depth(self):
		return self.depth

	def get_label(self):
		return self.label

	

#  A function to get the most common label of the previous tree. Reference - https://stackoverflow.com/questions/27755828/get-final-elements-of-nested-dictionary-in-python

def vals(x):
		if isinstance(x, dict):
				result = []
				for v in x.values():
						result.extend(vals(v))
				return result
		else:
				return [x]

def predict_core(inst, tree):
	for node in tree.keys():
		prediction = 0
		value = inst[node]
		try:
			tree = tree[node][value]
			if type(tree) is dict:
				prediction = predict_core(inst, tree)
			else:
				prediction = tree
		except KeyError:
				prediction = mode(vals(tree))     #  If some test example is a case that the trained tree never seen before and therefore doesn't have a branch for then for that test example it will predict that it's label is the most common label of the previous node of the tree.
			 
	return prediction

#  Calculating the average error of the actual and predicted labels

def predict(df, tree):
	y_predict = []
	for i in range(len(df)):
		inst = df.iloc[i,:]
		prediction = predict_core(inst, tree)
		y_predict.append(prediction)
	error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_predict) if x == y) / len(df[df.columns[-1]])
	return error


def predict_train(df, tree, actual_label):
  y_predict = []
  loss = []

  # pred_t = df.to_numpy()
  # pred_t_n = np.apply_along_axis(predict_core, 0, pred_t, tree)
  # y_predict = np.where(pred_t_n < 0, -1, 1)

  for i in range(len(df)):
    inst = df.iloc[i,:]
    prediction = predict_core(inst, tree)
    if prediction>0:
      prediction = 1
    elif prediction<0:
      prediction = -1
    else:
      prediction = 0
    y_predict.append(prediction)
  error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_predict) if x == y) / len(df[df.columns[-1]])

  error = sum(1 - np.isin(actual_label, y_predict))/y_predict.size

  length = len(y_predict)
  for i in range(0, length):              # STEP 2
    if actual_label[i] != y_predict[i]:
        loss.append(1)
    else:
        loss.append(0)

  # loss = 1 - np.isin(actual_label, y_predict)
  return error, y_predict, loss


class ID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10, subset=2):
		self.option = option
		self.max_depth = max_depth
		self.subset = subset
	
	
	def set_max_depth(self, max_depth):
		self.max_depth = max_depth


	def set_option(self, option):
		self.option = option

	def calc_ME(self, data, label_dict):
		
		pass

	def calc_entropy(self, data, label_dict):
		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		entropy = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			if p != 0:
				entropy += -p * math.log2(p)
		return entropy
	
	def calc_ME(self, data, label_dict):
		
		pass
		
	
	def calc_GI(self, data, label_dict):
		
		
		pass
	

	def get_majority_label(self,column):
		

		majority_label = column.value_counts().idxmax()

		return majority_label

	def get_heuristics(self):

		if self.option == 0:
			heuristics = self.calc_entropy
		if self.option == 1:
			heuristics = self.calc_ME
		if self.option == 2:
			heuristics = self.calc_GI

		return heuristics


	def get_feature_with_max_gain(self, data, label_dict, features_dict, sampled_features):

		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict)

		max_gain = float('-inf')
		max_f_name = ''

		for f_name in sampled_features:
			gain = 0
			f_values = features_dict[f_name]
			for val in f_values:
				subset = data[data[f_name] == val]
				p = len(subset) / len(data)
				
				gain += p * heuristics(subset, label_dict)

			# get maximum gain and feature name	
			gain = measure - gain
			if gain > max_gain:
				max_gain = gain
				max_f_name = f_name

		return max_f_name
		

	def best_feature_split(self, cur_node):
		next_nodes = []
		features_dict = cur_node['features_dict']
		label_dict = cur_node['label_dict']
		dt_node = cur_node['dt_node']
		data = cur_node['data']

		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		
		if len(data) > 0:
			majority_label = self.get_majority_label(data[label_key])
			
		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict)

		# check leaf nodes
		if measure == 0 or dt_node.get_depth() == self.max_depth or len(features_dict) == 0:
			dt_node.set_leaf(True)
			if len(data) > 0:
				dt_node.set_label(majority_label)
			return next_nodes

		
		children = {}

		# randomly select features 
		keys = list(features_dict.keys())

		if len(keys) > self.subset:
			sampled_features = np.random.choice(keys, self.subset, replace=False)
		else:
			sampled_features = keys 

		max_f_name = self.get_feature_with_max_gain(data, label_dict, features_dict, sampled_features)
		dt_node.set_feature(max_f_name)

		# remove the feature that has been splitted on, get remaining features
		rf = copy.deepcopy(features_dict)
		rf.pop(max_f_name, None)
	
		for val in features_dict[max_f_name]:
			child_node = TNode()
			child_node.set_label(majority_label)
			child_node.set_depth(dt_node.get_depth() + 1)
			children[val] = child_node
			primary_node = {'data': data[data[max_f_name] == val],'features_dict': rf, 'label_dict': label_dict, 'dt_node': child_node}
			next_nodes.append(primary_node)
		
		# set chiildren nodes
		dt_node.set_children(children)
		
		return next_nodes
	   
	
	# construct the decision tree
	def construct_dt(self, data, features_dict, label_dict):

		# bfs using queue
		import queue
		dt_root = TNode()
		dt_root.set_depth(0)
		root = {'data': data,'features_dict': features_dict, 'label_dict': label_dict, 'dt_node': dt_root}

		Q = queue.Queue()
		Q.put(root)
		while not Q.empty():
			cur_node = Q.get()
			for node in self.best_feature_split(cur_node):
				Q.put(node)
		return dt_root
	

	def classify_one(self, dt, data):
		temp = dt
		while not temp.is_leaf(): 
			temp = temp.children[data[temp.feature]]
		return temp.label

	def predict(self, dt, data):
		return data.apply(lambda row: self.classify_one(dt, row), axis=1)

# processing training data to convert numerical values into catagorical using median

def proccess_train_for_numerical_value(df):     # returns modified training data and the median value
	df_o = df.select_dtypes(include='object')
	df_n = df.select_dtypes(include=[np.int64])
	b = pd.DataFrame()
	m = {}
	for d in df_n.columns:
		med = np.median(df_n[d])
		m[d] = med
		b[d] = df_n[d]>med
	pd_new = pd.concat([df_o, b], axis=1)
	pd_new = pd_new[list(df.columns)]
	
	return pd_new, m

# processing training data to convert numerical values into catagorical using median of the TRAINING SET

def proccess_test_for_numerical_value(df, train_m):     # returns modified training data and the median value
	df_o = df.select_dtypes(include='object')
	df_n = df.select_dtypes(include=[np.int64])
	b = pd.DataFrame()
	for d in df_n.columns:
		# med = np.median(df_n[c])
		# m[d] = med
		b[d] = df_n[d]>train_m[d]
	pd_new = pd.concat([df_o, b], axis=1)
	pd_new = pd_new[list(df.columns)]
	
	return pd_new

def category_to_numerical_features(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 

def get_bank_data():

	column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
	types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
			'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

	# load train data 
	train_data =  pd.read_csv('data/bank/train.csv', names=column_names, dtype=types)
	# load test data 
	test_data =  pd.read_csv('data/bank/test.csv', names=column_names, dtype=types)

	numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

	train_data = category_to_numerical_features(train_data, numerical_features)
	test_data = category_to_numerical_features(test_data, numerical_features)

	features_dict = {}
	features_dict['age'] = [0, 1]
	features_dict['balance'] = [0, 1]
	features_dict['day'] = [0, 1]
	features_dict['previous'] = [0, 1]
	features_dict['campaign'] = [0, 1]
	features_dict['pdays'] = [0, 1]
	features_dict['duration'] = [0, 1]
	features_dict['loan'] = ['yes', 'no']
	features_dict['default'] = ['yes', 'no']
	features_dict['housing'] = ['yes', 'no']
	features_dict['job'] = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services']
	features_dict['marital'] = ['married','divorced','single']
	features_dict['education'] = ['unknown', 'secondary', 'primary', 'tertiary']
	features_dict['contact'] = ['unknown', 'telephone', 'cellular']
	features_dict['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
	features_dict['poutcome'] = ['unknown', 'other', 'failure', 'success']
	
	label_dict = {}
	label_dict['y'] = ['yes', 'no']

	return features_dict, label_dict, train_data, test_data

