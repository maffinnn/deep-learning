import numpy as np

# Fully Connected Layer
class FCLayer(object):

    def __init__(self, input_size, output_size, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            input_size: size of each input sample
            output_size: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.input_size = input_size
        self.output_size = output_size
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input):

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation(Wx+b) to Input, and return results.

        ############################################################################
        # input.shape = (batch_size, features)
        # print("layer.Input.shape:", Input.shape)
        self.Input = Input
        # self.W.shape = (features, 10)
        # self.b.shape = (1, 10)
        result = np.matmul(Input, self.W) + self.b
        # (batch_size, 10)
        return result
        

    def backward(self, delta):
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient using the later layer's gradient: delta
        # self.grad_W and self.grad_b will be used in optimizer.py
        
        # delta.shape = (batch_size, 10)
        # input.shape = (batch_size, features)
        # self.grad_W.shape = (features£¬ 10)
        self.grad_W = np.transpose(np.matmul(np.transpose(delta), self.Input)/self.Input.shape[0])
        self.grad_b = np.average(delta, axis=0)
        
        

    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.input_size + self.output_size))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.input_size, self.output_size))
        self.b = np.random.normal(0, init_std, (1, self.output_size))
