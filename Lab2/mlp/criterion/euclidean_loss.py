""" Euclidean Distance Loss """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.
    
	def forward(self, logit, gt):
		"""
	      parameters: (minibatch)
	      - logit: output from the last fully-connected layer, shape=(batch_size, 10)
	      - gt: ground truth, shape=(batch_size, 10)
	    """

		############################################################################
	    # TODO: 
		# compute the current minibatch's average accuracy rate and loss and save into self.accu and self.loss respectively(used in solver.py)
		# return only self.loss
		self.gt = gt
		self.logit = logit
		predictions = np.argmax(self.logit, axis=1)
		labels = np.where(gt == 1)[1]
		losses = np.sum((self.gt-self.logit)**2, axis=1)/2
		self.loss = np.mean(losses)
		self.acc = len(np.where(predictions == labels)[0])/len(labels)
	    ############################################################################
		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# compute gradient and return the result(same shape as logit)
		delta = self.logit - self.gt
		return delta
	    ############################################################################