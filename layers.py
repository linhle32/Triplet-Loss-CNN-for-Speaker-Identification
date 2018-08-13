#####
#definitions for deep layers
#   embedding/fully-connected layer
#   convolutional layer (includes pooling)
#####

#import libraries
#set GPU
import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32,exception_verbosity='high'"
#import theano
from theano import *
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
#others
import numpy as np

####
#class for embedding layer
#input: theano variable 
#n_in: input shape
#n_out: output shape
#alpha: alpha for triplet loss if the layer is output layer
#W: weight matrix, theano variable
#b: bias vector, theano variable
#activation: theano function
class EmbeddingLayer(object):
    
    def __init__(self,input,n_in,n_out,alpha=0,W=None,b=None,activation=T.nnet.relu):
        #parameters
        self.activation = activation
        self.alpha = alpha
        #randomize weights if not given
        if W is None:
            W = theano.shared(
                value=np.random.normal(size=(n_in,n_out),loc=0,scale=0.01).astype(dtype=theano.config.floatX),
                name="W",
                borrow=True
            )
        self.W = W
        #set bias to 0 vector if not given
        if b is None:
            b = theano.shared(
                value=np.zeros((n_out),dtype=theano.config.floatX),
                name="b",
                borrow=True
            )
        self.b = b
        #layer input
        self.input = input
        #layer output
        self.output = self.activation(T.dot(self.x,self.W) + self.b)
        #add trainable parameters
        self.params = [self.W, self.b]
    
    #function to compute triplet loss
    #trp is a list of indexes for [reference_index, positive_index, negative_index
    def TripletLoss(self,trp):
        return T.mean(T.sum((self.output[trp[:,0]] - self.output[trp[:,1]])**2,axis=1) + self.alpha - \
                      T.sum((self.output[trp[:,0]] - self.output[trp[:,2]])**2,axis=1))

    
####
#class for CNN layer
#input: theano variable for data
#filter_shape: list of [n_output_filters, n_input_filters, filter_height, filter_width]
#image_shape: list of [batch_size, n_channels, image_height, image_width
#poolsize: pooling height, pooling width
class LeNetConvPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2), W=None,b=None,activation=T.nnet.relu):
        
        self.input = input
        #initialize weights if not given
        if W is None:
            W = theano.shared(
                numpy.asarray(
                    np.random.normal(size=filter_shape,loc=0,scale=0.01).astype(dtype=theano.config.floatX),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W = W
        #initialize bias if not given
        if b is None:
            b = theano.shared(value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True)
        self.b = b
        #convolution computation
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        #pooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        #CNN blockoutput
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #layer parameters
        self.params = [self.W, self.b]