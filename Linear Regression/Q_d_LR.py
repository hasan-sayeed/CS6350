import numpy as np 
import math
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv

#  BGD functions

def cost_function(X, Y, W):
	res = 0 
	for i in range(len(X)):
		temp = (Y[i] - np.dot(W, X[i]))**2 
		res += temp 
	return 0.5*res


def LMS_gradiant(X, Y, r):
	costs = []  

	W = np.zeros(X.shape[1])


	e = math.inf

	while e > 10e-6:
		grad_w = np.zeros(X.shape[1])
		
		for j in range(len(X[0])):
			temp = 0 
			for i in range(len(X)):
				temp += X[i][j] *(Y[i] - np.dot(W, X[i]))
			grad_w[j] = temp 

		new_W = W + r*grad_w

		e = LA.norm(W - new_W)
		costs.append(cost_function(X, Y, W))

		W = new_W

	costs.append(cost_function(X, Y, W))
	return W, costs


#  SGD functions

def sgd(X, Y, r):

	W = np.zeros(X.shape[1])


	e = math.inf

	costs = [cost_function(X, Y, W)]

	while e > 10e-10:
		i = random.randrange(len(X))

		grad_w = np.zeros(X.shape[1])
		for j in range(len(X[0])): 
			grad_w[j] = X[i][j] *(Y[i] - np.dot(W, X[i]))

		new_W = W + r*grad_w
		W = new_W
		new_cost = cost_function(X, Y, W) 
		e = abs(new_cost - costs[-1])
		costs.append(new_cost)

	return W, costs


# train data
train = np.loadtxt('data/concrete/train.csv', delimiter =',',usecols = range(8))
#test data    
test = np.loadtxt('data/concrete/test.csv', delimiter =',',usecols = range(8))

# get vector x and y for both train and test datasets
X_train = train[:,:-1]
one_train = np.ones(X_train.shape[0])
D_train = np.column_stack((one_train, X_train))
Y_train = train[:,-1]

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((one_test, X_test))
Y_test = test[:,-1]

print()
# part a 
print("________________ Part 4(a) ________________")
print("Batch gradient descent")

r = 0.01
W, costs = LMS_gradiant(D_train, Y_train, r)
test_cost_value = cost_function(D_test, Y_test, W)
print("Learning rate: ", r)
print("The learned weight vector: ", W)
print("Test data cost function value: ", test_cost_value)
fig1 = plt.figure()
plt.plot(costs, c= 'r')
fig1.suptitle('Batch Gradient Descent ', fontsize=17)
plt.xlabel('iteration', fontsize=16)
plt.ylabel('Cost Function', fontsize=16)
# plt.show()
# fig1.savefig("BGD_cost_function.png")
# print("Figure has been saved!")

# part b
print()
print()
print("________________ Part 4(b) ________________")
print("stochastic gradient descent")

r = 0.001
W, costs = sgd(D_train, Y_train, r)
test_cost_value = cost_function(D_test, Y_test, W)
print("Learning rate: ", r)
print("The learned weight vector: ", W)
print("Test data cost function value: ", test_cost_value)
fig2 = plt.figure()
plt.plot(costs, c = 'r')
fig2.suptitle('Stochastic Gradient Descent ', fontsize=17)
plt.xlabel('iteration', fontsize=16)
plt.ylabel('Cost Function', fontsize=16)

# fig2.savefig("SGD_cost_function.png")
# print("The figure has been saved! ")

# part c
print()
print()
print("________________ Part 4(c) ________________")
print("Optimal weight vector with analytical form")

new_D_train = D_train.T
temp = np.matmul(new_D_train, new_D_train.T)
invtemp = inv(temp)
final_w = np.matmul(np.matmul(invtemp, new_D_train), Y_train)
test_cost_value = cost_function(D_test, Y_test, final_w)
print("The learned weight vector: ", final_w)
print("Test data cost function value: ", test_cost_value)

plt.show()