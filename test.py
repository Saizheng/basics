from basics.basic import *
from basics.util import *
import pdb

def main():
    #states_Px_mean = {'layers' : [Layer(W_init(3, 2), b_init(2), 'tanh'),
    #                              Layer(W_init(2, 2), b_init(2), 'tanh')],}
    states_Px_mean = {'sizes_activations' : [[3, 2, 2], ['tanh', 'sigm']]} 
    mlp = MLP(states_Px_mean)
    X = T.matrix()
    func = theano.function([X], mlp.f(X))
    y = func([[1,1, 1], [2,2, 2]])

    pdb.set_trace()
    print mlp.layers[0].W
    mlp = func(1)
    print mlp.layers[0].W

if __name__ == "__main__":
    main()

