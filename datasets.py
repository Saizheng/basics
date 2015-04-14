__author__ = "Saizheng Zhang"

import theano, cPickle 
import theano.tensor as T
import numpy as np
from util import *

class dataset_MNIST(object):
    def __init__(self, path):
        (self.train_x, self.train_y), \
        (self.valid_x, self.valid_y), \
        (self.test_x, self.test_y) \
            = np.load(path)

    # DC centering and contrast normalization
    def pre(self, data):
        N = data.shape[0]
        data_ = data 
        # preprocessing
        for i in xrange(N):
            #DC centering
            P = data_[i] - data_[i].mean()
            #contrast normalization
            P = P/np.sqrt(np.sum(P*P)+0.01)
            data_[i] = P 
        return data_

    # numpy form
    def np(self):
        return self.train_x, self.train_y, \
               self.valid_x, self.valid_y, \
               self.test_x, self.test_y

    # one-hot
    def shared(self, onehot_flag = False):
        if onehot_flag:
            return sharedX(self.train_x), sharedX(one_hot(self.train_y)), \
                   sharedX(self.valid_x), sharedX(one_hot(self.valid_y)), \
                   sharedX(self.test_x), sharedX(one_hot(self.test_y))
        else:
            return sharedX(self.train_x), sharedX(self.train_y), \
                   sharedX(self.valid_x), sharedX(self.valid_y), \
                   sharedX(self.test_x), sharedX(self.test_y)
