import torch.nn as nn
import torch
import random


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):    
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection 
        IoU = (intersection + self.smooth)/(union + self.smooth)
        return 1. - IoU


class BCELoss(nn.Module):
    def __init__(self):
      super(BCELoss, self).__init__()
      self.seed = 42
  
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        # sampling since the dataset is biased
        random.seed(self.seed)
        indexes = random.sample([i for i in range(len(y_pred))], k=int(0.5*len(y_pred)))
        y_pred = y_pred[indexes]
        y_true = y_true[indexes]
        return -(y_pred.log()*y_true + (1-y_true)*(1-y_pred).log()).mean()



