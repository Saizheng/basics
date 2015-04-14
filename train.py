import theano, time, pickle, os
import theano.tensor as T
import numpy as np
from util import *
import pdb

def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

class MainLoop(object):
    def __init__(self, data, model, configs, **kwargs):
        self.data = data
        # model at least includes model.X, model.Y, model[cost], model[err], model[params] 
        self.model = model

        # Load basic training configs
        self.epoch = configs['epoch']
        self.batchsize = configs['batchsize']
        self.n_batches = self.data[0][0].get_value().shape[0]/self.batchsize #TODO:more elegant
        self.update_method = configs['update_method']
        self.update_settings = configs['update_settings']
        self.write_model = False
        if configs.has_key('write_model'):
            self.write_model = configs['write_model']
            self.write_model_path = configs['write_model_path'] 

    def run(self):

        i, e = T.iscalar(), T.fscalar() 
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.data

        # updates
        updates = self.update_method(self, *self.update_settings)
        #updates = sgd(self.model.cost, self.model.params, 0.05)

        # training givens
        givens_train = lambda i : { self.model.X : train_x[ i*self.batchsize : (i+1)*self.batchsize ],
                                    self.model.Y : train_y[ i*self.batchsize : (i+1)*self.batchsize ] }
        givens_valid = { self.model.X : valid_x, self.model.Y : valid_y }
        givens_test = { self.model.X : test_x,  self.model.Y : test_y}
 
        # training and testing function
        train_sync = theano.function([i,e], [self.model.cost], givens = givens_train(i), on_unused_input='ignore',
                                     updates = updates)
        valid_cost = theano.function([], [self.model.err], on_unused_input='ignore', givens={self.model.X:valid_x, self.model.Y:valid_y}  )
        test_cost = theano.function([], [self.model.err], on_unused_input='ignore', givens={self.model.X:test_x, self.model.Y:test_y}  )
    
        # training loop
        print 'epochs  train_cost  valid_err  test_err  test_best  time'    
        min_test_err = 100
        t = time.time();
        monitor = { 'train' : [], 'valid' : [], 'test' : []}
        for e in range(1, self.epoch+1) :
            monitor['train'].append(  np.array([ train_sync(i,e) for i in range(self.n_batches) ]).mean(axis=0)  )
            if e % 1 == 0 :
                temp_test_err = test_cost()
                monitor['valid'].append(valid_cost())
                monitor['test'].append(temp_test_err)
                if monitor['test'][-1][0] < min_test_err:
                    min_test_err =  monitor['test'][-1][0]
                print e, monitor['train'][-1][0], monitor['valid'][-1][0], monitor['test'][-1][0], \
                      min_test_err, str(time.time() - t) +'seconds'#
            if self.write_model == True: 
                output = open(self.write_model_path, 'wb')
                pickle.dump(self.model, output)
                output.close()

def update_sgd(mainloop, lr, momentum):
    setattr(mainloop,
            'velocitys',
            [sharedX(param.get_value() * .0)
             for param in mainloop.model.params])
    grads = T.grad(mainloop.model.cost, mainloop.model.params)
    update_velocitys = [momentum * v - lr * grad for v, grad in
                        zip(mainloop.velocitys, grads)]
    updates = [(param, param + up_v) for param, up_v in
               zip(mainloop.model.params, update_velocitys)]
    updates.extend([(v, up_v) for v, up_v in
                    zip(mainloop.velocitys, update_velocitys)])
    return updates

def update_rmsprop(mainLoop, lr):
    pass
