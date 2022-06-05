import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.V_W = 0
        self.V_b = 0

    def step(self):
        """One backpropagation step, update weights layer by layer"""

        layers = self.model.layerList
        for i, layer in enumerate(layers):
            if layer.trainable:

                ############################################################################
                # TODO: Put your code here
                # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.

                # In the momentum learning algorithm, assume unit mass
                # The larger learning rate is relative to epsilon, the more previous gradients affect the current direction
                # Weight update with momentum
                self.V_W = self.momentum * self.V_W - self.learning_rate * layer.grad_W
                layer.W += self.V_W
                self.V_b = self.momentum * self.V_b - self.learning_rate * layer.grad_b
                layer.b += self.V_b
                # # Weight update without momentum
                # g
                # layer.W += -self.learning_rate * layer.grad_W
                # layer.b += -self.learning_rate * layer.grad_b

                ############################################################################







