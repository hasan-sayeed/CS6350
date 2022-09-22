import numpy as np

# Entropy of the dataset

def entropy_of_total(df):
  label = df.keys()[-1]
  counts = df[label].unique()    # unique labels
  total_entropy = 0
  for count in counts:
    probability = df[label].value_counts()[count]/len(df[label])
    total_entropy += -probability * np.log2(probability)
  return np.float64(total_entropy)

# Entropy of attribute

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

# Information gain calculation and best attribute

def best_attribute(df):
  IG = []
  for key in df.keys()[:-1]:
    IG.append(entropy_of_total(df) - entropy_of_attribute(df, key))
  return df.keys()[:-1][np.argmax(IG)]

# Updated dataframe

def updated_dataframe(df, attribute, val):
  return df[df[attribute] == val].reset_index(drop = True)

# construct the tree

def construct_tree(df, tree = None):
  node = best_attribute(df)               # Get the best attribute first
  att_counts = np.unique(df[node])
  label = df.keys()[-1]
  if tree is None:
    tree = {}
    tree[node] = {}
  for att_count in att_counts:           # Go to each branch
    updated_data = updated_dataframe(df,node,att_count)
    label_vals, label_counts = np.unique(updated_data[label], return_counts = True)
    if len(label_counts) == 1:
      tree[node][att_count] = label_vals[0]
    else:
      tree[node][att_count] = construct_tree(updated_data)
  return tree

def predict(inst, tree):
  for node in tree.keys():
    value = inst[node]
    tree = tree[node][value]
    prediction = 0
    if type(tree) is dict:
      prediction = predict(inst, tree)
    else:
      prediction = tree
  return prediction