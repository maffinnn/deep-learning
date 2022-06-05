""" Softmax Cross Entropy """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape (batch_size, 10)
	      - gt: the ground truth label, shape (batch_size, 10)
	    """

		############################################################################
	    # TODO: 
		# Calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
		self.gt = gt
		self.proba = np.exp(logit)/(EPS+np.sum(np.exp(logit),axis=1)[:,np.newaxis])
		predictions = np.argmax(self.proba, axis=1)
		labels = np.where(gt == 1)[1]
		losses = -np.sum(np.multiply(self.gt, np.log(self.proba)), axis=1)
		self.loss = np.mean(losses)
		self.acc = len(np.where(predictions == labels)[0])/len(labels)
	    ############################################################################
		return self.loss


	def backward(self):

		############################################################################
	    # TODO: 
        # Calculate and return the gradient (have the same shape as logits)
		delta = self.proba - self.gt
		return delta
	    ############################################################################
