import numpy as np
import torch
import torch.nn as nn

class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def __add__(self, other):
        return CombinedLoss(self, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)
    
    def __mul__(self, other):
        return ProductLoss(self, other)

    def __rmul__(self, other):
        return self if other == 0 else self.__add__(other)


class CombinedLoss(Loss):
    def __init__(self, *losses):
        super().__init__()
        self.losses = losses

    def forward(self, *args, **kwargs):
        total = 0
        for loss_fn in self.losses:
            total += loss_fn(*args, **kwargs)
        return total
    
class ProductLoss(Loss):
    def __init__(self, *losses):
        super().__init__()
        self.losses = losses

    def forward(self, *args, **kwargs):
        total = 0
        for loss_fn in self.losses:
            total *= loss_fn(*args, **kwargs)
        return total
    
    
class WeightedMSE(Loss):
    def __init__(self, power, weight = 1.0):
        super().__init__()
        self.power = power
        self.weight = weight

    def forward(self, predicted, targets, weights, *args, **kwargs):
        # Example: weighted mean squared error
        loss = weights * torch.pow((predicted - targets),self.power)
        return self.weight * torch.sum(loss) #/ torch.sum(weights)
    
class EntropyLoss(Loss):

    def __init__(self, power = 1.0, weight = 1.0):
        super().__init__()
        self.weight = weight
        self.power = power

    def forward(self, predicted, targets, weights, *args, **kwargs):
        error = torch.pow(predicted - targets, self.power)
        loss = - torch.sum(error * torch.log(1 - weights + 1e-8))
        return self.weight * loss
    
class SurprisalLoss(Loss):

    def __init__(self, power = 1.0, weight = 1.0):
        super().__init__()
        self.weight = weight
        self.power = power

    def forward(self, predicted, targets, weights, *args, **kwargs):
        error = torch.pow(predicted - targets, self.power)
        # surprisal = -torch.log(weights + 1e-8)
        surprisal = -torch.log(weights + 1e-8)
        necessity = torch.log(1.0 - weights + 1e-8)
        # loss = torch.sum(error * (necessity + surprisal - weights) / (necessity + surprisal - weights).sum(dim=1,keepdim = True))
        loss = torch.sum(error * torch.max(surprisal,dim=1).values)
        # loss *= torch.sum(error * torch.max(surprisal,dim=1,keepdim=True).values)
        # loss += torch.sum(error * weights)
        # loss = torch.sum(error * (necessity + surprisal - weights) / (necessity + surprisal - weights).sum(dim=1,keepdim = True))
        # loss = torch.sum(error * (necessity + surprisal))
        return self.weight * loss
    
class NecessityLoss(Loss):

    def __init__(self, power = 1.0, weight = 1.0):
        super().__init__()
        self.weight = weight
        self.power = power

    def forward(self, predicted, targets, weights, *args, **kwargs):
        error = torch.pow(predicted - targets, self.power)
        necessity = -torch.log(1.0 - weights + 1e-8)
        loss = (error * necessity).sum(dim=1).mean() 
        return self.weight * loss

