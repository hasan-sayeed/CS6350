# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:23:17 2018
@author: Kaai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Here we make the functions that can calculate the gradiant based on the
# solution dJ/dw_j = -sum_{i=1}^m (y_i-w^T x_i)x_{ij} (pg 31 lecture 8)

def batch_gradiant(weights, x, y):
    '''
    Used in the batch gradient decent function. 
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated 
        from len([b]+[w]) where b is the bias andd w is the weights.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.
    Returns
    ------------
    gradiant: list,
        This list contains the gradiant [dJ/dw_j, dJ/dw_j+1, ....., dJ/dw_N].
        (Optimized weights should produce a vector of zeros [0, 0, ...., 0])
    '''
    gradiant = []
    for j in range(0, weights[0,:].shape[1]):
        grad_j = 0
        for i in range(0, y[0,:].shape[1]):
            grad_j += (y[0,i] - weights*x[i].transpose())*x[i,j]
        gradiant.append(-grad_j[0,0])
    return gradiant

def predict(weights, x_i):
    '''
    Function should be used to predict label with trained weights
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1 calulated 
        from len([b]+[w]) where b is the bias andd w is the weights. Trained
        weights should produce results similar to the label.
        
    x_i: list,
        List of size (1 x N-1) N is the number of features + 1. This should be
        a single feature vector assocaited with one instance of data.
    Returns
    ------------
    prediction: float,
        Returnss the expected label calculated from the model weights and 
        features.
    '''
    x_i = np.matrix(x_i)
    x_i = np.hstack((np.ones((1,1)), x_i))
    prediction = weights*x_i.transpose()
    return prediction[0,0]

def lms_error(weights, x, y):
    '''
    Used in the batch and stochastic gradient decent functions.
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weight.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.
    Returns
    ------------
    error: float,
        This return the lms error for the entire dataset. This is done by
        summing the squared difference between the prediction and the label
        values for each instance.
    '''
    error = 0
    for i in range(0, y[0,:].shape[1]):
        error += 0.5 * (y[0,i]-weights*x[i].transpose())**2
    return error[0, 0]

def batch_gradient_descent(weights, x, y, error_threshold=0.5,
                           learning_rate=0.1, max_iterations=500 , tolerance=1e-6):
    '''
    Function to perform batch gradiant descent on linear data. 
    Parameters
    ------------
    weights: list,
        List of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weights.
    x: list,
        List of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
    y: list,
        List of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.
    
    error_threshold: float,
        Set the stopping error for the model.
    learning_rate: float,
        This is the rate at which the gradiant is subtracted from the weights 
        during optimization. Larger numbers can allow for faster convergence, 
        but may also result in ocsillations. Smaller values allow for closer 
        convergence to actual minimum.
    max_iterations: int,
        Set the upper limit for how many iterations the program will run. This
        is useful when the minimum-error can't be reached, or the learning rate
        is unstable or too slow.
        
    tolerance: float,
        Set the minimum error progress that needs to be made for the model to 
        continue
    Returns
    ------------
    weights: float,
        This returns the optimized weights using the batch gradiaent decent 
        technique.
    converged: boolean,
        True for convergence, False if max_iterations reached
    iterations: list,
        List containing iteration numbers
    errors: list,
        list containing error associated with iterations
    '''
    weights = np.matrix(weights)
    x = np.matrix(x)
    x = np.hstack((np.ones((len(y), 1)), x))
    y = np.matrix(y)
    error = lms_error(weights, x, y)
    count = 0
    iterations = []
    errors = []
    converged = True
    while error >= error_threshold:
        if count >= max_iterations:
            converged = False
            break
        weights = weights - learning_rate * np.array(batch_gradiant(weights, x, y))
        epsilon = np.linalg.norm(learning_rate * np.array(batch_gradiant(weights, x, y)))
        if epsilon <= tolerance:
            converged = True
            break
        if lms_error(weights, x, y) > 10*error:
            converged = False
            break
        error = lms_error(weights, x, y)
        iterations.append(count)
        errors.append(error)
        count += 1
    return [weights, converged, iterations, errors, epsilon]


def stochastic_gradiant(weights, x, y, i):
    '''
    Used in the stochastic gradient decent function. 
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated 
        from len([b]+[w]) where b is the bias andd w is the weights.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.
    i: integer,
        used to keep track of the data instance for which we are performing our 
        calculation.
        
    Returns
    ------------
    gradiant: list,
        This list contains the gradiant [dJ/dw_j, dJ/dw_j+1, ....., dJ/dw_N].
        (Optimized weights should produce a vector of zeros [0, 0, ...., 0])
    '''
    gradiant = []
    weights = np.matrix(weights)
    for j in range(0, weights[0,:].shape[1]):
        grad_j = -(y[0, i] - weights*x[i].transpose())*x[i, j]
        gradiant.append(grad_j[0,0])
    return gradiant

def stochastic_gradiant_descent(weights, x, y, error_threshold=0.5,
                                learning_rate=0.01, max_iterations=500, print_text=False):
    '''
    Function to perform batch gradiant descent on linear data. 
    Parameters
    ------------
    weights: list,
        List of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weights.
    x: list,
        List of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
    y: list,
        List of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.
    
    error_threshold: float,
        Set the stopping error for the model.
    learning_rate: float,
        This is the rate at which the gradiant is subtracted from the weights 
        during optimization. Larger numbers can allow for faster convergence, 
        but may also result in ocsillations. Smaller values allow for closer 
        convergence to actual minimum.
    max_iterations: int:
        Set the upper limit for how many iterations the program will run. This
        is useful when the minimum-error can't be reached, or the learning rate
        is unstable or too slow.
    Returns
    ------------
    weights: float,
        This returns the optimized weights using the batch gradiaent decent 
        technique.
    iterations: list,
        List containing iteration numbers
    errors: list,
        list containing error associated with iterations
    '''
    weights = np.array(weights)
    x = np.matrix(x)
    x = np.hstack((np.ones((len(y),1)), x))
    y = np.matrix(y)
    error = lms_error(weights, x, y)
    count = 0
    iterations = []
    errors = []
    while error >= error_threshold:
        for i in range(0, y[0,:].shape[1]):
            if count >= max_iterations:
                return [weights, iterations, errors]
            if print_text == True:
                print('Feature:', x[i])
                print('w:', weights, 'gradiant:', stochastic_gradiant(weights, x, y, i))
            weights = weights - learning_rate * np.array(stochastic_gradiant(weights, x, y, i))
            if print_text == True:
                print('w_t:', weights)
            if lms_error(weights, x, y) > 10*error:
                for n in range(0,10):
                    print('ERROR!!!')
                print('Unstable parameters, reduce learning rate')
                return
            error = lms_error(weights, x, y)
            iterations.append(count)
            errors.append(error)
            count += 1
    return [weights, iterations, errors]

# %%
# read in the data we want to work with. In this case it comes from:
# https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test

df_test = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/concrete/test.csv', header=None)
df_train = pd.read_csv('D:/Fall_2022/ML/HWs/hw_2/CS6350/Ensemble Learning/data/concrete/train.csv', header=None)

# we need to separate the features from the labels (targets). From the data
# description, we know the first columns contain features, and the final column
# contains the label/target

def convert_dataframe_to_list(df):
    df_list = []
    df_ = df.transpose()
    for column_index in df_:
        df_list.append(list(df_[column_index]))
    return df_list

columns = df_test.columns.values
X_train = convert_dataframe_to_list(df_train[columns[:-1]])
y_train = list(df_train[columns[-1]])
weights = [0] + list(np.zeros(len(X_train[0])))

X_test = convert_dataframe_to_list(df_test[columns[:-1]])
y_test = list(df_test[columns[-1]])


# %%
# =============================================================================
#                                    (a)
# =============================================================================
converged = False
r = 1
loop_weights = weights

print('\n\nfinding max r of values (2^-n)\n-------------------')
while converged == False:
    batch = batch_gradient_descent(loop_weights, 
                                   X_train, y_train,
                                   error_threshold=1,
                                   learning_rate= r,
                                   max_iterations=10,
                                   tolerance=1e-2)
    converged = batch[1]
    r = r/2
    print('epsilon:', batch[4], 'r:', r)
    loop_weights = batch[0]
# %%

r = 0.005
batch = batch_gradient_descent(weights, 
                                   X_train, y_train,
                                   error_threshold=1,
                                   learning_rate= r,
                                   max_iterations=500,
                                   tolerance=1e-2)
iteration = batch[2]
errors = batch[3]
trained_batch_weights = batch[0]
epsilon = batch[4]

plt.figure(1, figsize=(4, 4))
font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 18}
plt.rc('font', **font)
plt.plot(iteration, errors, 'rx')
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Error', fontsize=22)
plt.xlim((0, max(iteration)))
plt.ylim((12, max(errors)))
plt.xticks(np.linspace(0, max(iteration), 2))
plt.yticks((12, errors[-1], max(errors)))
plt.legend(['Batch Descent'], loc='best')
plt.draw()

#print(epsilon)
## %%
#y_predictions = []
#for X in X_train:
#    y_prediction = predict(trained_batch_weights, X)
#    y_predictions.append(y_prediction)
#
#plt.figure(2, figsize=(8, 8))
#font = {'family' : 'DejaVu Sans',
#    'weight' : 'normal',
#    'size'   : 18}
#plt.rc('font', **font)
#plt.plot(y_train, y_predictions, 'ro')
#plt.plot([0, 1000], [0, 1000], 'k-')
#max_value = max(y_test)
#plt.xlabel('Actual', fontsize=22)
#plt.ylabel('Predicted', fontsize=22)
#plt.xlim((0, max_value))
#plt.ylim((0, max_value))
#ticks = np.linspace(0, max_value, 5)
#plt.xticks(ticks)
#plt.yticks(ticks)
#plt.legend(['Batch Descent'], loc='best')
#plt.show()
#
## %%
#y_predictions = []
#for X in X_test:
#    y_prediction = predict(trained_batch_weights, X)
#    y_predictions.append(y_prediction)
#
#
#plt.figure(3, figsize=(8, 8))
#font = {'family' : 'DejaVu Sans',
#    'weight' : 'normal',
#    'size'   : 18}
#plt.rc('font', **font)
#plt.plot(y_test, y_predictions, 'ro')
#plt.plot([0, 1000], [0, 1000], 'k-')
#max_value = max(y_test)
#plt.xlabel('Actual', fontsize=22)
#plt.ylabel('Predicted', fontsize=22)
#plt.xlim((0, max_value))
#plt.ylim((0, max_value))
#ticks = np.linspace(0, max_value, 5)
#plt.xticks(ticks)
#plt.yticks(ticks)
#plt.legend(['Density (MPDB)','Ideal Performance'], loc='best')
#plt.show()

# %%
#calculate cost function for test data
def get_error(weights, x, y):
    x = np.matrix(x)
    x = np.hstack((np.ones((len(y),1)), x))
    y = np.matrix(y)
    cost = lms_error(weights, x, y)
    return cost

get_error(trained_batch_weights, X_train, y_train)
print('\nbatch weight:', trained_batch_weights)
print('test error for batch decent:', get_error(trained_batch_weights, X_test, y_test))

# %%
# =============================================================================
#                                    (B)
# =============================================================================
r = 0.01
stochastic = stochastic_gradiant_descent(weights, 
                                   X_train, y_train,
                                   error_threshold=1,
                                   learning_rate= r,
                                   max_iterations=500,
                                   print_text=False)

iteration = stochastic[1]
errors = stochastic[2]
trained_stochastic_weights = stochastic[0]
#epsilon = stochastic[4]

plt.figure(2, figsize=(4, 4))
font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 18}
plt.rc('font', **font)
plt.plot(iteration, errors, 'ro')
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Error', fontsize=22)
plt.xlim((0, max(iteration)))
plt.ylim((0, max(errors)))
plt.xticks(np.linspace(0, max(iteration), 2))
plt.yticks((0, errors[-1], max(errors)))
plt.legend(['Stochastic Descent'], loc='best')
plt.draw()

get_error(trained_stochastic_weights, X_train, y_train)
print('\nstochastic weight:', trained_stochastic_weights)
print('test error for stochastic decent:', get_error(trained_stochastic_weights, X_test, y_test))

# %%
# =============================================================================
#                                    (c)
# =============================================================================

X_correct = []
for instance in X_train:
    X_correct.append([1]+instance)
    
X = np.matrix(X_correct).transpose()
Y = np.matrix(y_train).transpose()

XX = X*X.transpose()
XY = X*Y
inverse = np.linalg.inv(XX)

w = inverse*XY

weight = np.array(w[:,0]).ravel()
optimized_weights = np.matrix(weight)

error = lms_error(weight, np.matrix(X_correct), np.matrix(y_train))
print('\nanalytic solution, weights:', weight)



get_error(optimized_weights, X_train, y_train)
print('analytic solution, error:', get_error(optimized_weights, X_test, y_test))

plt.show()

## %%
#y_predictions = []
#for X in X_train:
#    y_prediction = predict(weight, X)
#    y_predictions.append(y_prediction)
#
#plt.figure(2, figsize=(8, 8))
#font = {'family' : 'DejaVu Sans',
#    'weight' : 'normal',
#    'size'   : 18}
#plt.rc('font', **font)
#plt.plot(y_train, y_predictions, 'ro')
#plt.plot([0, 1000], [0, 1000], 'k-')
#max_value = max(y_test)
#plt.xlabel('Actual', fontsize=22)
#plt.ylabel('Predicted', fontsize=22)
#plt.xlim((0, max_value))
#plt.ylim((0, max_value))
#ticks = np.linspace(0, max_value, 5)
#plt.xticks(ticks)
#plt.yticks(ticks)
#plt.legend(['Density (MPDB)','Ideal Performance'], loc='best')
#plt.show()
#
## %%
#y_predictions = []
#for X in X_test:
#    y_prediction = predict(weight, X)
#    y_predictions.append(y_prediction)
#
#
#plt.figure(3, figsize=(8, 8))
#font = {'family' : 'DejaVu Sans',
#    'weight' : 'normal',
#    'size'   : 18}
#plt.rc('font', **font)
#plt.plot(y_test, y_predictions, 'ro')
#plt.plot([0, 1000], [0, 1000], 'k-')
#max_value = max(y_test)
#plt.xlabel('Actual', fontsize=22)
#plt.ylabel('Predicted', fontsize=22)
#plt.xlim((0, max_value))
#plt.ylim((0, max_value))
#ticks = np.linspace(0, max_value, 5)
#plt.xticks(ticks)
#plt.yticks(ticks)
#plt.legend(['Density (MPDB)','Ideal Performance'], loc='best')
#plt.show()