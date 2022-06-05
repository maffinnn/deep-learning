import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):
    
    def forward(self, logits, labels):
        """
          Inputs: (minibatch)
          - logits: forward results from the last FCLayer, shape (batch_size, 10)
          - labels: the ground truth label, shape (batch_size, )
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for backward
        
        # convert to one-hot encoding for labels
        self.onehot_labels = np.zeros(shape=logits.shape)
        self.onehot_labels[np.arange(labels.shape[0]),labels] = 1
        self.proba = np.exp(logits)/(np.sum(np.exp(logits),axis=1)[:,np.newaxis])
        predictions = np.argmax(self.proba, axis=1)
        losses = -np.sum(np.multiply(self.onehot_labels, np.log(self.proba)), axis=1)
        loss = np.sum(losses)/len(labels)
        acc = len(np.where(predictions == labels)[0])/len(labels)
        return loss, acc

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logits)
        delta = self.proba - self.onehot_labels
        return delta
        


