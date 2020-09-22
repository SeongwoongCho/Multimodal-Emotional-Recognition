import random
import os
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def to_onehot(label,num_classes=7):
    return np.eye(num_classes)[label]

def cross_entropy():
    def _cross_entropy(input, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = nn.LogSoftmax()
        if size_average:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
    return _cross_entropy
