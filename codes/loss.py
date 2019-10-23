from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        loss = np.sum((input - target) ** 2)
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        grad = input - target
        return grad


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        pass

    def backward(self, input, target):
        '''Your codes here'''
        pass
