""" Fully-Connected Layer """

import numpy as np

class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		"""
		Apply linear transformation to the input: y = Wx + b
		Parameters:
			num_input: input size
			num_output: output size
			actFunction: specifies activation function 
			trainable: true if it is training set
		"""
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output))
		self.grad_b = np.zeros((1, num_output))


	def forward(self, Input):

		############################################################################
	    # TODO: 
		# Apply linear transformation y = Wx + b and return the result
		self.Input = Input
		return np.matmul(Input, self.W) + self.b

	    ############################################################################


	def backward(self, delta):
		# delta is computed in the later layer
		############################################################################
	    # TODO: 
		# compute gradient according to delta
		self.grad_W = np.transpose(np.matmul(np.transpose(delta), self.Input)/self.Input.shape[0])
		self.grad_b = np.average(delta, axis=0)
		delta = np.matmul(delta, np.transpose(self.W))
		return delta
	    ############################################################################

	def XavierInit(self):
		# Initialization
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
		self.b = np.random.normal(0, init_std, (1, self.num_output))
