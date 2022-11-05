import numpy as np
import pandas as pd
from perceptron_functions import Standerd_Perceptron, Voted_Perceptron, Averaged_Perceptron

train_data = np.loadtxt('Perceptron/bank-note/train.csv', delimiter=',',usecols = range(5))
test_data = np.loadtxt('Perceptron/bank-note/test.csv', delimiter=',',usecols = range(5))


std = Standerd_Perceptron(train_data, test_data, r = 0.01, T = 10)
print('________2(a)________')
print('Standerd Perceptron:')
print('Learned weight vector: {},\ntest error: {}'.format(std.weights, std.test_error))

vt = Voted_Perceptron(train_data, test_data, r = 0.01, T = 10)
weight_df = pd.DataFrame(vt.w_list, columns=['w1', 'w2', 'w3', 'w4', 'b'])
weight_df['counts'] = pd.DataFrame(vt.C_m_list)
print('\n\n________2(b)________')
print('Voted Perceptron:')
print('Distinct weight vectors and counts: \n')
print(weight_df)
print('\ntest error: {}'.format(vt.test_error))

av = Averaged_Perceptron(train_data, test_data, r = 0.01, T = 10)
print('\n\n________2(c)________')
print('Averaged Perceptron:')
print('Learned weight vector: {},\ntest error: {}'.format(av.a, av.test_error))