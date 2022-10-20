import pandas as pd
import numpy as np
from statistics import mode
from id3_functions_ensamble import id3, predict_train, predict


#  First weight
def first_w(df):
  w = [1/len(df)]*len(df)
  df['label'] = df['label']*w
  return df, w

def decision_stump(df_train, df_test, actual_label):
  tree= id3(df_train, metric = 'entropy', tree_depth = 1)
  train_error, train_pred, loss = predict_train(df_train, tree, actual_label)
  test_error, test_pred = predict(df_test, tree)
  return train_pred, test_pred, loss, train_error, test_error

# total error, epsilon
def epsilon(loss, w):
  epsilon = sum([x*y for x,y in zip(loss, w)])
  return epsilon

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