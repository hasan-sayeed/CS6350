import numpy as np
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Entropy of the dataset

def entropy_of_total(df):
  label = df.keys()[-1]
  counts = df[label].unique()    # unique labels
  total_entropy = 0
  for count in counts:
    probability = df[label].value_counts()[count]/len(df[label])
    total_entropy += -probability * np.log2(probability)
  return np.float64(total_entropy)

# Entropy of attributes

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

# Majority error of the dataset

def me_of_total(df):
  label = df.keys()[-1]
  label_vals, label_counts = np.unique(df[label], return_counts = True)   # unique labels
  majority_value_count = np.amax(label_counts)
  probability = majority_value_count/len(df[label])
  # for label_val in label_vals:
  #   probability = df[label].value_counts()[label_val]/len(df[label])
  #   total_me += -probability * np.log2(probability)
  total_me = 1 - probability
  return np.float64(total_me)

# Majority error of attributes

def me_of_attribute(df, attribute):
  label = df.keys()[-1]
  attr_vals = df[attribute].unique()
  # label_vals, label_counts = np.unique(df[label][df[attribute] == attr_val], return_counts = True)
  me = 0
  for attr_val in attr_vals:
    attr_label_vals, attr_label_counts = np.unique(df[label][df[attribute] == attr_val], return_counts = True)
    majority_value_count = np.amax(attr_label_counts)
    # num = len(df[attribute][df[attribute] == attr_val][df[label] == label_val])
    den = len(df[attribute][df[attribute] == attr_val])
    probability = majority_value_count/den
    me_tmp = 1 - probability
    # for label_val in attr_label_vals:
    #   num = len(df[attribute][df[attribute] == attr_val][df[label] == label_val])
    #   den = len(df[attribute][df[attribute] == attr_val])
    #   probability = num/den
    #   me_tmp += -probability * np.log2(probability + 0.000001)
    me += (den/len(df))*me_tmp
  return np.float64(me)

# Gini index of the dataset

def gi_of_total(df):
  label = df.keys()[-1]
  counts = df[label].unique()    # unique labels
  total_gi = 1
  for count in counts:
    probability = df[label].value_counts()[count]/len(df[label])
    total_gi += -probability**2
  return np.float64(total_gi)

# Gini index of attributes

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

# Information gain calculation and best attribute

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

# Updated dataframe

def updated_dataframe(df, attribute, val):
  return df[df[attribute] == val].reset_index(drop = True)

# construct the tree

def id3(df, tree = None, metric = 'entropy', tree_depth= 1000):
  
  '''
  Parameters
        ----------
        metric : str, optional
            Which varient of information gain you want to use. Possible values are
            "entropy", "majority error" and "gini index", by default "entropy.
        tree_depth : str, optional
            Maximum tree depth you want, by default 1000.
  '''

  node = best_attribute(df, metric)               # Get the best attribute first
  att_counts = np.unique(df[node])
  label = df.keys()[-1]
  t_d = 0
  if tree is None:
    tree = {}
    tree[node] = {}
  for att_count in att_counts:           # Go to each branch
    updated_data = updated_dataframe(df,node,att_count)
    label_vals, label_counts = np.unique(updated_data[label], return_counts = True)
    if len(label_counts) == 1:
      tree[node][att_count] = label_vals[0]
    else:
      t_d += 1
      # tree[node][att_count] = construct_tree(updated_data)
      if t_d<tree_depth:
        tree[node][att_count] = id3(updated_data)
      elif t_d==tree_depth:
        majority_label = np.where(label_counts == np.amax(label_counts))
        tree[node][att_count] = label_vals[majority_label[0][0]]
      
  return tree#, metric, tree_depth

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
        prediction = mode(vals(tree))
       
  return prediction

def predict(df, tree):
  y_predict = []
  for i in range(len(df)):
    inst = df.iloc[i,:]
    prediction = predict_core(inst, tree)
    y_predict.append(prediction)
  accuracy = metrics.accuracy_score(df[df.columns[-1]], y_predict)
  return accuracy

# def id3(df, metric = 'entropy', tree_depth = 1000):
  
#   '''
#   Parameters
#         ----------
#         metric : str, optional
#             Which varient of information gain you want to use. Possible values are
#             "entropy", "majority error" and "gini index", by default "entropy.
#         tree_depth : str, optional
#             Maximum tree depth you want, by default 1000.
#   '''

#   tree = construct_tree(df, metric, tree_depth)

  
#   return tree