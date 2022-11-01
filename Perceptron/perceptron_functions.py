import numpy as np

class Standerd_Perceptron:
	
	def __init__(self, train_data, test_data, r, T):
		self.train_data = train_data
		self.test_data = test_data
		self.r = r
		self.T = T
		self.weights = self.fit()
		self.train_error = self.predict(self.train_data)
		self.test_error = self.predict(self.test_data)
	
	
	def process_data(self, data):
		X_original = data[:, :-1]
		y_original = data[:,-1]

		# notational sugar
		augmented_column = np.ones(X_original.shape[0])
		X = np.column_stack((X_original, augmented_column))

		# convert label to 1 and -1
		y = 2*y_original - 1

		return X, y


	def fit(self):

		X , y = self.process_data(self.train_data)
		n_features = X.shape[1]
		n_examples = X.shape[0]
		indices = np.arange(n_examples)

		# initialize weight vector
		self.weights = np.zeros(n_features)

		for _ in range(self.T):
			np.random.shuffle(indices)
			X = X[indices, :]
			y = y[indices]
			for j in range(n_examples):
				pred = np.dot(self.weights, X[j])
				if y[j]*pred <= 0:
					self.weights = self.weights + self.r*y[j]*X[j]

		return self.weights

	def predict(self, data):
		X , y = self.process_data(data)
		w_reshaped = np.reshape(self.weights, (-1, 1))
		pred_list = np.matmul(X, w_reshaped)
		pred_list[pred_list<=0] = -1
		pred_list[pred_list>0] = 1

		ground_truth = np.reshape(y, (-1, 1))
		
		error = sum(abs(pred_list - ground_truth)/2)/len(pred_list)
		
		return error

	
class Voted_Perceptron:
	
	def __init__(self, train_data, test_data, r, T):
		self.train_data = train_data
		self.test_data = test_data
		self.r = r
		self.T = T
		self.w_list, self.C_m_list = self.fit()
		self.train_error = self.predict(self.train_data)
		self.test_error = self.predict(self.test_data)
	
	
	def process_data(self, data):
		X_original = data[:, :-1]
		y_original = data[:,-1]

		# notational sugar
		augmented_column = np.ones(X_original.shape[0])
		X = np.column_stack((X_original, augmented_column))

		# convert label to 1 and -1
		y = 2*y_original - 1

		return X, y


	def fit(self):

		X , y = self.process_data(self.train_data)
		n_features = X.shape[1]
		n_examples = X.shape[0]
		indices = np.arange(n_examples)

		# initialize weight vector
		w = np.zeros(n_features)

		# initialize list of weights and C_m
		C_m = 0
		w_list = np.array([])
		C_m_list = np.array([])

		for _ in range(self.T):
			np.random.shuffle(indices)
			X = X[indices, :]
			y = y[indices]
			for r in range(n_examples):
				pred = np.dot(w, X[r])
				if y[r]*pred <= 0:
					w_list = np.append(w_list, w)
					C_m_list = np.append(C_m_list, C_m)
					w = w + self.r*y[r]*X[r]
					C_m = 1
				else:
					C_m = C_m + 1
		k = C_m_list.shape[0]
		w_list = np.reshape(w_list, (k,-1))

		return w_list, C_m_list

	def predict(self, data):
		X , y = self.process_data(data)
		C_m_list_reshaped = np.reshape(self.C_m_list, (-1, 1))

		# transpose w_list as it is a kXm dimensional matix.
		w_list_t = np.transpose(self.w_list)
		pred_list = np.matmul(X, w_list_t)   # pred_list is a nXk dimensional matrix.
		pred_list[pred_list<=0] = -1
		pred_list[pred_list>0] = 1

		voted_pred_list = np.matmul(pred_list, C_m_list_reshaped)  # C_m_list is a kX1 dimensional vector here.
		voted_pred_list[voted_pred_list<=0] = -1
		voted_pred_list[voted_pred_list>0] = 1

		ground_truth = np.reshape(y, (-1, 1))
		
		error = sum(abs(voted_pred_list - ground_truth)/2)/len(pred_list)
		
		return error


class Averaged_Perceptron:
	
	def __init__(self, train_data, test_data, r, T):
		self.train_data = train_data
		self.test_data = test_data
		self.r = r
		self.T = T
		self.a = self.fit()
		self.train_error = self.predict(self.train_data)
		self.test_error = self.predict(self.test_data)
	
	
	def process_data(self, data):
		X_original = data[:, :-1]
		y_original = data[:,-1]

		# notational sugar
		augmented_column = np.ones(X_original.shape[0])
		X = np.column_stack((X_original, augmented_column))

		# convert label to 1 and -1
		y = 2*y_original - 1

		return X, y


	def fit(self):

		X , y = self.process_data(self.train_data)
		n_features = X.shape[1]
		n_examples = X.shape[0]
		indices = np.arange(n_examples)

		# initialize weight vector
		w = np.zeros(n_features)

		# initialize averaged weight vector "a"
		a = np.zeros(n_features)

		for _ in range(self.T):
			np.random.shuffle(indices)
			X = X[indices, :]
			y = y[indices]
			for j in range(n_examples):
				pred = np.dot(w, X[j])
				if y[j]*pred <= 0:
					w = w + self.r*y[j]*X[j]
				a = a + w

		return a

	def predict(self, data):
		X , y = self.process_data(data)
		a_reshaped = np.reshape(self.a, (-1, 1))
		pred_list = np.matmul(X, a_reshaped)
		pred_list[pred_list<=0] = -1
		pred_list[pred_list>0] = 1

		ground_truth = np.reshape(y, (-1, 1))
		
		error = sum(abs(pred_list - ground_truth)/2)/len(pred_list)
		
		return error




# def process_data(data):
# 	X_original = data[:, :-1]
# 	y_original = data[:,-1]

# 	# notational sugar
# 	augmented_column = np.ones(X_original.shape[0])
# 	X = np.column_stack((X_original, augmented_column))

# 	# convert label to 1 and -1
# 	y = 2*y_original - 1

# 	return X, y










# def averaged_perceptron(data, r, T):

# 	X , y = process_data(data)
# 	n_features = X.shape[1]
# 	n_examples = X.shape[0]
# 	indices = np.arange(n_examples)

# 	# initialize weight vector
# 	w = np.zeros(n_features)

# 	# initialize averaged weight vector "a"
# 	a = np.zeros(n_features)

# 	for _ in range(T):
# 		np.random.shuffle(indices)
# 		X = X[indices, :]
# 		y = y[indices]
# 		for j in range(n_examples):
# 			pred = np.dot(w, X[j])
# 			if y[j]*pred <= 0:
# 				w = w + r*y[j]*X[j]
# 			a = a + w

# 	return a


# def avg_perceptron_predict(data, a):
# 	X , y = process_data(data)
# 	a = np.reshape(a, (-1, 1))
# 	pred_list = np.matmul(X, a)
# 	pred_list[pred_list<=0] = -1
# 	pred_list[pred_list>0] = 1

# 	ground_truth = np.reshape(y, (-1, 1))
	
# 	error = sum(abs(pred_list - ground_truth)/2)/len(pred_list)
	
# 	return error



	# def process_data(data):
	# X_original = data[:, :-1]
	# y_original = data[:,-1]

	# # notational sugar
	# augmented_column = np.ones(X_original.shape[0])
	# X = np.column_stack((X_original, augmented_column))

	# # convert label to 1 and -1
	# y = 2*y_original - 1

	# return X, y


# def standerd_perceptron(data, r, T):

# 	X , y = process_data(data)
# 	n_features = X.shape[1]
# 	n_examples = X.shape[0]
# 	indices = np.arange(n_examples)

# 	# initialize weight vector
# 	w = np.zeros(n_features)

# 	for _ in range(T):
# 		np.random.shuffle(indices)
# 		X = X[indices, :]
# 		y = y[indices]
# 		for j in range(n_examples):
# 			pred = np.dot(w, X[j])
# 			if y[j]*pred <= 0:
# 				w = w + r*y[j]*X[j]

# 	return w


# def std_perceptron_predict(data, w):
# 	X , y = process_data(data)
# 	w = np.reshape(w, (-1, 1))
# 	pred_list = np.matmul(X, w)
# 	pred_list[pred_list<=0] = -1
# 	pred_list[pred_list>0] = 1

# 	ground_truth = np.reshape(y, (-1, 1))
	
# 	error = sum(abs(pred_list - ground_truth)/2)/len(pred_list)
	
# 	return error


# def voted_perceptron(data, r, T):

# 	X , y = process_data(data)
# 	n_features = X.shape[1]
# 	n_examples = X.shape[0]
# 	indices = np.arange(n_examples)

# 	# initialize weight vector
# 	w = np.zeros(n_features)

# 	# initialize list of weights and C_m
# 	C_m = 0
# 	w_list = np.array([])
# 	C_m_list = np.array([])

# 	for _ in range(T):
# 		np.random.shuffle(indices)
# 		X = X[indices, :]
# 		y = y[indices]
# 		for j in range(n_examples):
# 			pred = np.dot(w, X[j])
# 			if y[j]*pred <= 0:
# 				w_list = np.append(w_list, w)
# 				C_m_list = np.append(C_m_list, C_m)
# 				w = w + r*y[j]*X[j]
# 				C_m = 1
# 			else:
# 				C_m = C_m + 1
# 	k = C_m_list.shape[0]
# 	w_list = np.reshape(w_list, (k,-1))

# 	return w_list, C_m_list



# def voted_perceptron_predict(data, w_list, C_m_list):
# 	X , y = process_data(data)
# 	C_m_list = np.reshape(C_m_list, (-1, 1))

# 	# transpose w_list as it is a kXm dimensional matix.
# 	w_list = np.transpose(w_list)
# 	pred_list = np.matmul(X, w_list)   # pred_list is a nXk dimensional matrix.
# 	pred_list[pred_list<=0] = -1
# 	pred_list[pred_list>0] = 1

# 	voted_pred_list = np.matmul(pred_list, C_m_list)  # C_m_list is a kX1 dimensional vector here.
# 	voted_pred_list[voted_pred_list<=0] = -1
# 	voted_pred_list[voted_pred_list>0] = 1

# 	ground_truth = np.reshape(y, (-1, 1))
	
# 	error = sum(abs(voted_pred_list - ground_truth)/2)/len(pred_list)
	
# 	return error