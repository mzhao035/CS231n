import numpy as np


class NearestNeighbor():
	def __init__(self):
		self.xtr = None
		self.ytr = None

	def train(self, x, y):
		"""x is N*D where each row is an example. y is 1-dimension of size N."""
		# the nearest neighbor classifier simply remember all the training data.
		self.xtr = x
		self.ytr = y

	def predict(self, x):
		"""x is N*D where each row is an example we wish to predict label for."""
		num_test = x.shape[0]
		# make sure that the output type matches the input type.
		ypred = np.zeros(num_test, dtype=self.ytr.dtype)

		# loop over all test rows
		for i in range(num_test):
			distances = np.sum(np.abs(self.xtr - x[i, :]), axis=-1)
			min_index = np.argmin(distances)
			ypred[i] = self.ytr[min_index]

		return ypred



