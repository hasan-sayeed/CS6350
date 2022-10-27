

#  Sources used: lecture slides, github, stackoverflow, towards data science articles 


import math
import copy
import queue
import numpy as np
import pandas as pd
from statistics import mode

# Entropy of the whole dataset

def entropy_of_total(df):
  targets = df.to_numpy()[:,-1]
  p = sum(np.where(targets>0, targets, 0))
  n = sum(-np.where(targets<0, targets, 0))
  probability_p = p/(p+n)
  probability_n = n/(p+n)
  total_entropy = -probability_p * np.log2(probability_p) - probability_n * np.log2(probability_n)
  return np.float64(total_entropy)

# Entropy of a single attributes

def entropy_of_attribute(df, attribute):
  attr_vals = df[attribute].unique()
  targets = df.to_numpy()[:,-1]
  p = sum(np.where(targets>0, targets, 0))
  n = sum(-np.where(targets<0, targets, 0))
  entropy = 0
  for attr_val in attr_vals:
    att_index = df.columns.get_loc(attribute)
    target_att = df.to_numpy()
    target_att_vals = target_att[np.where((target_att[:,att_index]==attr_val))][:,-1]
    p_a = sum(np.where(target_att_vals>0, target_att_vals, 0))
    n_a = sum(-np.where(target_att_vals<0, target_att_vals, 0))
    entropy_tmp = 0
    probability_p = p_a/(p_a + n_a)
    probability_n = n_a/(p_a + n_a)
    entropy_tmp += -probability_p * np.log2(probability_p + 0.000001) - probability_n * np.log2(probability_n + 0.000001)  
    entropy += (p_a + n_a)/(p + n)*entropy_tmp
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

def id3(df, metric, tree_depth, tree = None, t_d = 1):
  
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

def predict(df, tree):
  y_predict = []
  for i in range(len(df)):
    inst = df.iloc[i,:]
    prediction = predict_core(inst, tree)
    y_predict.append(prediction)
  error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_predict) if x == y) / len(df[df.columns[-1]])

  # pred_t = df.to_numpy()
  # y_predict = np.apply_along_axis(predict_core, 0, pred_t, tree)
  # # y_predict = np.where(pred_t_n < 0, -1, 1)
  
  # error = sum(1 - np.isin(pred_t[:,-1], y_predict))/y_predict.size
  error = 1 - sum(1 for x,y in zip(df[df.columns[-1]],y_predict) if x == y) / len(df[df.columns[-1]])

  return error, y_predict


class Mod_ID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10):
		self.option = option
		self.max_depth = max_depth
	
	
	def set_option(self, option):
		self.option = option

	def set_max_depth(self, max_depth):
		self.max_depth = max_depth


	

	def calc_ME(self, data, label_dict, weights):
		pass

	def calc_entropy(self, data, label_dict, weights):
		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		total = np.sum(weights)
		col = np.array(data[label_key].tolist())

		if total == 0:
			return 0
		entropy = 0

		for value in label_values:
			w = weights[col==value]
			p = np.sum(w) / total

			if p != 0:
				entropy += -p * math.log2(p)
		return entropy
	
	
	def calc_GI(self, data, label_dict):
		pass
	

	def get_heuristics(self):

		if self.option == 0:
			heuristics = self.calc_entropy
		if self.option == 1:
			heuristics = self.calc_ME
		if self.option == 2:
			heuristics = self.calc_GI

		return heuristics

	def get_majority_label(self, data, label_dict, weights):
		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]

		max_sum= float('-inf')
		col = np.array(data[label_key].tolist())

		for value in label_values:
			w = weights[col==value]
			w_sum = np.sum(w)
			if w_sum > max_sum:
				majority_label = value
				max_sum = w_sum

		return majority_label

	


	def get_feature_with_max_gain(self, data, label_dict, features_dict, weights):

		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict, weights)

		total = np.sum(weights)

		max_gain = float('-inf')
		max_f_name = ''

		for f_name, f_values in features_dict.items():
			col = np.array(data[f_name].tolist())
			gain = 0
			for val in f_values:
				w = weights[col==val]
				temp_weights = w 
				p = np.sum(temp_weights) /total
				subset = data[data[f_name] == val]
			
				gain += p * heuristics(subset, label_dict, temp_weights)


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
		weights = cur_node['weights']

		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		
		total = sum(weights)
		if total > 0:
			majority_label = self.get_majority_label(data, label_dict, weights)
			
		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict, weights)

		
		if measure == 0 or dt_node.get_depth() == self.max_depth or len(features_dict) == 0:
			dt_node.set_leaf(True)
			if total > 0:
				dt_node.set_label(majority_label)
			return next_nodes

		
		children = {}
		max_f_name = self.get_feature_with_max_gain(data, label_dict, features_dict, weights)
		dt_node.set_feature(max_f_name)

		# remove the feature that has been splitted on, get remaining features
		rf = copy.deepcopy(features_dict)
		rf.pop(max_f_name, None)
		
		col = np.array(data[max_f_name].tolist())

		for val in features_dict[max_f_name]:
			child_node = TNode()
			child_node.set_label(majority_label)
			child_node.set_depth(dt_node.get_depth() + 1)
			children[val] = child_node
			w = weights[col==val]
			primary_node = {'data': data[data[max_f_name] == val], 'weights': w, 'features_dict': rf, 'label_dict': label_dict, 'dt_node': child_node}
			next_nodes.append(primary_node)
		
		
		dt_node.set_children(children)
		
		return next_nodes
	   
	
	# constructing decision tree
	def construct_dt(self, data, features_dict, label_dict, weights):

		dt_root = TNode()
		dt_root.set_depth(0)
		root = {'data': data, 'weights':weights, 'features_dict': features_dict, 'label_dict': label_dict, 'dt_node': dt_root}

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
