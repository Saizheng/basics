__author__ = "Saizheng Zhang"

import theano
import theano.tensor as T
import numpy as np
from util import *

# W initialization
def W_init(n_in, n_out):
    #TODO: different kinds of initialization techniques 
    return rand_ortho((n_in, n_out), np.sqrt(6./(n_in +n_out)))

# b initialization
def b_init(n_out):
    return zeros((n_out,))

# basic layer class
class Layer(object):
    def __init__(self, W, b, activation = 'sigm'):
        assert W.get_value().shape[1] == b.get_value().shape[0]
        self.W = W
        self.b = b
        self.affine = False
        if activation is not 'linear':
            self.activation =eval(activation)
        else:
            self.affine = True

        self.n_in, self.n_out = self.W.shape
        self.param = [self.W, self.b]

    def f(self, x):
        if self.affine == True:
            return T.dot(x, self.W) + self.b
        else:
            return self.activation(T.dot(x, self.W) + self.b)

# basic MLP class
class MLP(object):
    def __init__(self, states):
        if states.has_key('model'):
            self.layers = states['model']
            self.params = []
            for i in xrange(len(self.layers)):
                self.params += self.layers[i].param
            self.layers_num = len(self.layers) 
        else:
            [sizes, activations] = states['config']
            assert len(sizes) == len(activations) + 1
            self.layers = []
            for i in xrange(len(sizes) - 1):
                self.layers.append(Layer(W_init(sizes[i], sizes[i+1]),
                                         b_init(sizes[i+1]),
                                         activations[i]))
            self.params = []
            for i in xrange(len(self.layers)):
                self.params += self.layers[i].param
            self.layers_num = len(self.layers)
 
    def f(self, x):
        y = x
        for i in xrange(self.layers_num):
            y = self.layers[i].f(y)
        return y

