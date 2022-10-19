import pandas as pd
import numpy as np
from statistics import mode
from id3_functions import id3, predict_train, predict


#  First weight
def first_w(df):
  first_w = [1/len(df)]*len(df)
  df['label'] = df['label']*first_w
  return df, first_w

def decision_stump(df_train, df_test, actual_label):
  tree= id3(df_train, metric = 'entropy', tree_depth = 1)
  train_error, pred, loss = predict_train(df_train, tree, actual_label)
  test_error = predict(df_test, tree)
  return pred, loss, train_error, test_error

# total error, epsilon
def epsilon(loss, first_weight):
  epsilon = sum([x*y for x,y in zip(loss,first_weight)])
  return epsilon

# alpha
def alpha(ep):
  alpha = (np.log((1 - ep)/ep))/2
  return alpha

#  Next weight
def next_w(df, alph, pred, actual_label, previous_w):
  w = previous_w*np.exp([-alph * i for i in [x*y for x,y in zip(pred,actual_label)]])
  w = w/sum(w)
  df['label'] = actual_label
  df['label'] = df['label']*w
  return w, df

def adaboost(df_train, df_test, actual_label, first_weight):
  pred, loss, decision_stump_trn_error, decision_stump_tst_error = decision_stump(df_train, df_test, actual_label)
  ep = epsilon(loss, first_weight)
  alph = alpha(ep)
  w, df_train = next_w(df_train, alph, pred, actual_label, first_weight)
  return pred, loss, ep, alph, w, df_train, decision_stump_trn_error, decision_stump_tst_error