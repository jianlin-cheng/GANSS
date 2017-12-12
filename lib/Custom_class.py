
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
import os 

class remove_1d_padding(Layer):
    
    """ 
    Remove the padding region in the softmax layer because the padding rows don't have the labels
    written by Jie Hou
    11/04/2017
    """
    
    # def __init__(self,  ktop=40, **kwargs):
    def __init__(self,  ktop, **kwargs):
        self.ktop = ktop
        super(remove_1d_padding, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.ktop,input_shape[2])
    
    def call(self,x,mask=None):
        output = x[:,0:self.ktop,:]
        return output
    
    def get_config(self):
        config = {'ktop': self.ktop}
        base_config = super(remove_1d_padding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class K_max_pooling1d(Layer):

    """ 
    Reshapes the Permuted CNN output so that it can be feed to the RNNs.
    Flattens the last two dims. 3D to 2D , (None,40,1) -> (1,40)
    """

   # def __init__(self,  ktop=40, **kwargs):
    def __init__(self,  ktop, **kwargs):
        self.ktop = ktop
        super(K_max_pooling1d, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.ktop,input_shape[2])

    def call(self,x,mask=None):
        output = x[T.arange(x.shape[0]).dimshuffle(0, "x", "x"),
              T.sort(T.argsort(x, axis=1)[:, -self.ktop:, :], axis=1),
              T.arange(x.shape[2]).dimshuffle("x", "x", 0)]
        return output

    def get_config(self):
        config = {'ktop': self.ktop}
        base_config = super(K_max_pooling1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

