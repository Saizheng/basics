__author__ = "Saizheng Zhang"

import theano, cPickle 
import theano.tensor as T
import numpy as np
from util import *
import pdb

class Dataset(object):
    def __init__(self):
        pass

    # open data patch
    def unpickle(self, file):
        fo = open(file, 'rb')
        dic = cPickle.load(fo)
        fo.close()
        return dic

    # preprocessing: DC centering and contrast normalization
    def pre(self, data):
        N = data.shape[0]
        data_ = data 
        # preprocessing
        for i in xrange(N):
            #DC centering
            P = data_[i] - data_[i].mean()
            #contrast normalization
            P = P/np.sqrt(np.sum(P*P)+1e-9)
            data_[i] = P 
        return data_

    # numpy form
    def np(self):
        return self.train_x, self.train_y, \
               self.valid_x, self.valid_y, \
               self.test_x, self.test_y

    # theano shared, one-hot/not one-hot
    def shared(self, onehot_flag = False):
        if onehot_flag:
            return sharedX(self.train_x), sharedX(one_hot(self.train_y)), \
                   sharedX(self.valid_x), sharedX(one_hot(self.valid_y)), \
                   sharedX(self.test_x), sharedX(one_hot(self.test_y))
        else:
            return sharedX(self.train_x), sharedX(self.train_y.astype('int32')), \
                   sharedX(self.valid_x), sharedX(self.valid_y.astype('int32')), \
                   sharedX(self.test_x), sharedX(self.test_y.astype('int32'))

    def shared_byclass(self, cls = [], change_label = True, onehot_flag = False):
        train_x, train_y, valid_x, valid_y, test_x, test_y = [],[],[],[],[],[]
        if len(cls) == 0:
            raise ValueError('cls cannot be an empty list!')
        else:
            for i in xrange(len(cls)):
                train_x.append(self.train_x[np.nonzero(self.train_y==cls[i])])
                valid_x.append(self.valid_x[np.nonzero(self.valid_y==cls[i])])
                test_x.append(self.test_x[np.nonzero(self.test_y==cls[i])])
                if change_label == True:
                    train_y.append(i*np.ones(self.train_y[np.nonzero(self.train_y==cls[i])].shape[0]))
                    valid_y.append(i*np.ones(self.valid_y[np.nonzero(self.valid_y==cls[i])].shape[0]))
                    test_y.append(i*np.ones(self.test_y[np.nonzero(self.test_y==cls[i])].shape[0]))
                else:
                    train_y.append(self.train_y[np.nonzero(self.train_y==cls[i])])
                    valid_y.append(self.valid_y[np.nonzero(self.valid_y==cls[i])])
                    test_y.append(self.test_y[np.nonzero(self.test_y==cls[i])])

        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        index_train = np.random.permutation(train_x.shape[0])
        train_x, train_y = train_x[index_train], train_y[index_train]

        valid_x = np.concatenate(valid_x)
        valid_y = np.concatenate(valid_y)
        index_valid = np.random.permutation(valid_x.shape[0])
        valid_x, valid_y = valid_x[index_valid], valid_y[index_valid]

        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)
        index_test = np.random.permutation(test_x.shape[0])
        test_x, test_y = test_x[index_test], test_y[index_test]

        if onehot_flag:
            return sharedX(train_x), sharedX(one_hot(train_y)), \
                   sharedX(valid_x), sharedX(one_hot(valid_y)), \
                   sharedX(test_x), sharedX(one_hot(test_y))
        else:
            return sharedX(train_x), sharedX(train_y.astype('int32')), \
                   sharedX(valid_x), sharedX(valid_y.astype('int32')), \
                   sharedX(test_x), sharedX(test_y.astype('int32'))



class Dataset_MNIST(Dataset):
    def __init__(self, path = '/data/lisatmp3/saizheng/problems/targetprop/mnist.pkl'):
        super(Dataset_MNIST, self).__init__()
        (self.train_x, self.train_y), \
        (self.valid_x, self.valid_y), \
        (self.test_x, self.test_y) \
            = np.load(path)

 
class Dataset_CIFAR10(Dataset):
    """
    CIFAR10 includes 5 training (valid) batch and 1 test batch,
    they are in the folder where user should claim in "path" in __init__.
    """
    # read training and test data from the folder named by path
    # path should end by "/"
    def __init__(self, path = '/data/lisa/data/cifar10/cifar-10-batches-py/', train_per = 0.9, color = False):
        
        # load training/validation patches and save them into np format
        train_x = []
        train_y = []
        for i in range(1, 6):
            dic = self.unpickle(path+'data_batch_'+str(i))
            train_x.append(dic['data'])
            train_y.append(dic['labels']) 
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        if color:
            #TODO: for color image, we need to find a way to do DC centering and contrast normalization
            self.train_x = train_x.reshape(50000, 3, 1024)[:50000*train_per]
            self.valid_x = train_y.reshape(50000, 3, 1024)[50000*train_per:]
        else:
            self.train_x = self.pre((train_x.reshape(50000, 3, 1024)*np.array([[[0.299], [0.587], [0.144]]])).sum(1)[:50000*train_per])
            self.valid_x = self.pre((train_x.reshape(50000, 3, 1024)*np.array([[[0.299], [0.587], [0.144]]])).sum(1)[50000*train_per:])

        self.train_y = train_y.reshape(50000)[:50000*train_per]
        self.valid_y = train_y.reshape(50000)[50000*train_per:]

        # load test patch and save them into np format
        test_x = []
        test_y = []
        dic = self.unpickle(path+'test_batch')
        test_x.append(dic['data'])
        test_y.append(dic['labels'])
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        if color:
            self.test_x = test_x.reshape(10000, 3, 1024)
        else:
            self.test_x = self.pre((test_x.reshape(10000, 3, 1024)*np.array([[[0.299], [0.587], [0.144]]])).sum(1))

        self.test_y = test_y.reshape(10000)


