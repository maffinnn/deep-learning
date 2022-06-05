"""Stochastic Gradient Descent"""
import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:

				############################################################################
			    # TODO:
				# update layer.grad_W and layer.grad_b with weightDecay 
				
				layer.W -= self.learningRate*(layer.grad_W + self.weightDecay*layer.W)
				layer.b -= self.learningRate*(layer.grad_b + self.weightDecay*layer.b)
				############################################################################
