""" ReLU """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU activation function: relu(x) = max(x, 0)
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# Apply ReLU on input and return the result
		self.Input = Input
		result = np.maximum(Input, np.zeros(shape=Input.shape))
		return result
	    ############################################################################


	def backward(self, delta):

		############################################################################
	    # TODO: 
		# compute gradient according to delta
		grad = np.zeros(shape = delta.shape)
		row_indices, col_indices = np.where(self.Input > 0)
		grad[row_indices, col_indices] = 1
		delta = np.multiply(delta, grad)
		return delta
	    ############################################################################
