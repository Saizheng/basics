__author__ = "Dong-Hyun Lee, Saizheng Zhang"

import theano
import theano.tensor as T
import numpy as np
import cPickle, time, os
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams


RNG = MRG_RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))

def castX(x) : return theano._asarray(x, dtype=theano.config.floatX)
def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def randn(shape,mean,std) : return sharedX( mean + std * np.random.standard_normal(size=shape) )
def randn_np(shape,mean,std) : return mean + std * np.random.standard_normal(size=shape)
def rand(shape, irange) : return sharedX( - irange + 2 * irange * np.random.rand(*shape) )
def zeros(shape) : return sharedX( np.zeros(shape) ) 
def ones(shape) : return sharedX(np.ones(shape))
def eye(shape) : return sharedX(np.eye(shape))

def rand_ortho(shape, irange) : 
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return sharedX(  np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V )) )

def rand_ortho_np(shape, irange) : 
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V ))

def one_hot(labels, nC=None):
    nC = np.max(labels) + 1 if nC is None else nC
    code = np.zeros( (len(labels), nC), dtype='float32' )
    for i,j in enumerate(labels) : code[i,j] = 1.
    return code

def T_one_hot(t, r=None):
    if r is None: r = T.max(t) + 1
    ranges = T.shape_padleft(T.arange(r), t.ndim)
    return T.cast( T.eq(ranges, T.shape_padright(t, 1)), 'floatX')

def sigm(x) : return T.nnet.sigmoid(x)
def sfmx(x) : return T.nnet.softmax(x)

def sfmx3(x) :
    result, updates = theano.scan(fn = lambda x_: sfmx(x_),
                                  outputs_info = None,
                                  sequences = [x])
    return result

def tanh(x) : return T.tanh(x)
def sign(x) : return T.switch(x > 0., 1., -1.)
def linear(x) : return x

def softplus(x) : return T.nnet.softplus(x)

#def relu(x) : return T.switch(x > 0., x, 0.)
def relu(x) : return x * (x > 1e-15)
def relu_(x) : return T.switch(x > 0., x, x/10.)
def reluz(x) : return T.switch(x>0., x, -x)
def relx(x) : return T.switch(x>0., reluz(x-1), reluz(x+1))

def reluzp(x, rng = RNG) :
    z = reluz(x)
    return z*rng.binomial(size = z.shape, p = sigm(z), dtype = theano.config.floatX)
# similar performance to relu
def relog(x) : return T.switch(x > 0., T.log(x+1), 0.)

# weight competition, it does not work well at this time though
def wcomp(W, x, ifprob = True):
    out = x.dimshuffle(0,1,'x')*W
    m = abs(out).max(axis=1)
    p = abs(out)/m.dimshuffle(0,'x',1)
    if ifprob:
        rand = RNG.uniform(p.shape, ndim=None, dtype=None, nstreams=None)
        mask = T.cast(rand < p, dtype = 'floatX' )
    else:
        mask = p
    return (out*mask).sum(axis=1)

def wgate(W, x):
    gate = 0.01#0.0001 
    out = x.dimshuffle(0,1,'x')*W
    return (out*(abs(out)>gate)).sum(axis=1)

#def negative_log_likelihood(probs, labels) : # labels are not one-hot code 
#    return - T.mean( T.log(probs)[T.arange(labels.shape[0]), T.cast(labels,'int32')] )

def NLL(probs, labels) : # labels are not one-hot code 
    return - T.mean( T.log(probs)[T.arange(labels.shape[0]), T.cast(labels,'int32')] )

def NLL_mul(probs, targets) :
    return - T.sum( targets * T.log(probs) ) / probs.shape[0]
    #return T.nnet.categorical_crossentropy(probs, targets).sum(axis=1).mean()

def NLL_bin(probs, targets) : 
    return - T.sum( targets * T.log(probs) + (1-targets) * T.log(1-probs) ) / targets.shape[0]
    return T.nnet.binary_crossentropy(probs, targets).sum(axis=1).mean()

def ce(probs, targets) : 
    return - targets * T.log(probs) + (1-targets) * T.log(1-probs)

def NLL_bin(probs, targets, weight = 1.) : 
    return - T.sum( weight * ( targets * T.log(probs) + (1-targets) * T.log(1-probs) ) ) / targets.shape[0]

def predict(probs) : return T.argmax(probs, axis=1) # predict labels from probs

def error(pred_labels,labels) : return 100.*T.mean(T.neq(pred_labels, labels)) # get error (%)

#def error(pred_labels,labels) : return 100.*T.mean(T.neq(pred_labels, labels)) # get error (%)

def mse(x,y, ax=1) : return T.sqr(x-y).sum(axis=ax).mean(axis=-1) # mean squared error

def se(x,y, ax=1) : return T.sqr(x-y).sum(axis=ax) # mean squared error
#def mce(p,t) : return T.nnet.binary_crossentropy( (p+1.001)/2.002, (t+1.001)/2.002 ).sum(axis=1).mean()

def mce(p,t) : return T.nnet.binary_crossentropy( p, t ).sum(axis=1).mean()




def samp_rect(x, offset=3):
    rand = RNG.uniform(x.shape, ndim=None, dtype=None, nstreams=None)
    return T.cast( rand < sigm(x - offset), dtype='floatX') * x

def samp(x) : # x in [0,1]
    rand = RNG.uniform(x.shape, ndim=None, dtype=None, nstreams=None)
    return T.cast( rand < x, dtype='floatX')

def gaussian(x, std, rng=RNG) : return x + rng.normal(std=std, size=x.shape, dtype=x.dtype)

def zero_mask(x,p,rng=RNG) :
    assert 0 <= p and p < 1
    return rng.binomial(p=1-p, size=x.shape, dtype=x.dtype) * x

def salt_pepper(x,p,rng=RNG) :
    assert 0 <= p and p < 1
    a = rng.binomial(p=1-p, size=x.shape, dtype=x.dtype)
    b = rng.binomial(p=0.5, size=x.shape, dtype=x.dtype)
    c = T.eq(a,0) * b
    return x*a + c    


def rms_prop( param_grad_dict, learning_rate, 
                    momentum=.9, averaging_coeff=.95, stabilizer=.0001) :
    updates = OrderedDict()
    for param in param_grad_dict.keys() :

        inc = sharedX(param.get_value() * 0.)
        avg_grad = sharedX(np.zeros_like(param.get_value()))
        avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

        new_avg_grad = averaging_coeff * avg_grad \
            + (1 - averaging_coeff) * param_grad_dict[param]
        new_avg_grad_sqr = averaging_coeff * avg_grad_sqr \
            + (1 - averaging_coeff) * param_grad_dict[param]**2

        normalized_grad = param_grad_dict[param] / \
                T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + stabilizer)
        updated_inc = momentum * inc - learning_rate * normalized_grad

        updates[avg_grad] = new_avg_grad
        updates[avg_grad_sqr] = new_avg_grad_sqr
        updates[inc] = updated_inc
        updates[param] = param + updated_inc

    return updates

import numpy

def get_ll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    lls = []
    for i in range(n_batches):
        begin = time.time()
        ll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        lls.extend(ll)

        #if i % 10 == 0:
        #    print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(lls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        lls.append(numpy.asarray(tmp).mean())

    ind = numpy.argmax(lls)
    return sigmas[ind]


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def DisplayNetwork(data, nx, ny, width) :
    i = 0
    data_ = np.zeros((nx*width, ny*width))
    for x in range(nx) :
        for y in range(ny) :
            data_[ x*width : (x+1)*width, y*width : (y+1)*width ] = data[i].reshape(width,width)#/data[i].min
            i += 1
    #plt.rcParams['figure.figsize'] = (16,16)
    plt.imshow(data_, cmap = cm.Greys_r)
    plt.savefig('t.png')

def visualize(n_start, n_end, filename = 'wgt.npy') :
    samples = np.load(filename)
    DisplayNetwork(samples[n_start:n_end], 10, 10, 28) 

