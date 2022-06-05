""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid Activation function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False
    

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# Apply Sigmoid on input and return the result
		self.Input = Input
		return 1/(1+np.exp(-Input))
	    ############################################################################

	def backward(self, delta):


		############################################################################
	    # TODO: 
		# compute gradient according to delta
		grad = np.exp(-self.Input)/(1+np.exp(-self.Input)**2)
		delta = np.multiply(delta, grad)
		return delta
	    ############################################################################
