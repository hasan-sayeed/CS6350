#  This work was done in collaboration with Ramsey Issa, PhD candidate MSE

#  Sources used: lecture slides, github, stackoverflow, towards data science articles 



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
