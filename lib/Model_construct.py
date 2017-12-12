# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2017

@author: Jie Hou
"""

"""
file: mnist_acgan.py
author: Luke de Oliveira (lukedeo@vaitech.io)

Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating, as
the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from collections import defaultdict
#import cPickle as pickle
import pickle
from PIL import Image

from six.moves import range
from Custom_class import remove_1d_padding


import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization


import numpy as np

np.random.seed(1337)

K.set_image_dim_ordering('th')

from Custom_class import remove_1d_padding


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(nb_filter, nb_row, subsample,use_bias=True):
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='relu', border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=2)(conv)
        return Activation("relu")(norm)
    
    return f




# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(nb_filter, nb_row, subsample,name,use_bias=True):
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='relu', border_mode="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(mode=0, axis=2,name="%s_nor" % name)(conv)
        return Dense(output_dim=3, init="he_normal",name="%s_softmax" % name, activation="softmax")(norm)
    
    return f


def build_generator(latent_size,AA_win,nb_filters,nb_layers,win_array,fea_num,n_class):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    
    #latent_size = 100
    #nb_filters = 10
    #nb_layers = 10
    #filter_sizes=[5,11,15,20]
    #win_array=[10]
    #AA_win = 15
    #fea_num = 20
    #nb_layers=10
    filter_sizes=win_array
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))
    # this will be our label
    sample_class = Input(shape=(1,), dtype='int32')
    
    # 10 classes in MNIST
    cls = Flatten()(Embedding(n_class, latent_size,
                              init='glorot_normal')(sample_class))
    
    DeepSS_convs = []
    
    # hadamard product between z-space and a class conditional embedding
    DeepSS_input_shape = [latent, sample_class]
    DeepSS_input = merge([latent, cls], mode='mul')
    DeepSS_input = Dense(1 * AA_win * fea_num, input_dim=latent_size, activation='relu')(DeepSS_input)
    DeepSS_input = Reshape((AA_win, fea_num))(DeepSS_input)
    
    for fsz in filter_sizes:
        DeepSS_conv = DeepSS_input
        for i in range(0,nb_layers):
            DeepSS_conv = _conv_bn_relu1D(nb_filter=nb_filters, nb_row=fsz, subsample=1)(DeepSS_conv)
        DeepSS_convs.append(DeepSS_conv)
    
    if len(filter_sizes)>1:
      DeepSS_fake_out = Merge(mode='average')(DeepSS_convs)
    else:
      DeepSS_fake_out = DeepSS_convs[0]  
    
    #DeepSS_CNN = Model(input=DeepSS_input_shape, output=DeepSS_fake_out)
    #DeepSS_CNN.summary()
    return Model(input=DeepSS_input_shape, output=DeepSS_fake_out)


def build_discriminator(AA_win,nb_filters,nb_layers,win_array,fea_num,n_class):
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    
    #latent_size = 100
    #nb_filters = 10
    #nb_layers = 10
    #filter_sizes=[5,11,15,20]
    #win_array=[10]
    #AA_win = 15
    #fea_num = 20
    #nb_layers=10
    
    DeepSS_input_shape =(AA_win,fea_num)
    #nb_filters = 10
    #nb_layers = 10
    #filter_sizes=[5,11,15,20]
    filter_sizes=win_array
    DeepSS_input = Input(shape=DeepSS_input_shape)
    DeepSS_convs = []
    for fsz in filter_sizes:
        DeepSS_conv = DeepSS_input
        for i in range(0,nb_layers):
            DeepSS_conv = _conv_bn_relu1D(nb_filter=nb_filters, nb_row=fsz, subsample=1)(DeepSS_conv)
        #DeepSS_conv = remove_1d_padding(ktop=ktop_node)(DeepSS_conv) ## remove the padding rows because they don't have targets
        #no need here, because if target is 0, the cross-entropy is zero, error will be not passed
        DeepSS_conv = Flatten()(DeepSS_conv)
        DeepSS_convs.append(DeepSS_conv)
    
    if len(filter_sizes)>1:
        DeepSS_flatten_out = Merge(mode='average')(DeepSS_convs)
    else:
        DeepSS_flatten_out = DeepSS_convs[0]  
    
    
    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(DeepSS_flatten_out)
    aux = Dense(n_class, activation='softmax', name='auxiliary')(DeepSS_flatten_out)
    
    #DeepSS_CNN = Model(input=[DeepSS_input], output=[fake, aux])
    #DeepSS_CNN.summary()
    return Model(input=[DeepSS_input], output=[fake, aux])



def DeepCov_SS_with_paras(win_array,feature_num,use_bias,hidden_type,nb_filters,nb_layers,opt):
    #### for model 40~100
    #feature_num = 40
    #ktop_node= ktop_node
    print "Setting hidden models as ",hidden_type
    ########################################## set up ss model
    DeepSS_input_shape =(None,feature_num)
    #nb_filters = 10
    #nb_layers = 10
    #filter_sizes=[5,11,15,20]
    filter_sizes=win_array
    DeepSS_input = Input(shape=DeepSS_input_shape)
    DeepSS_convs = []
    for fsz in filter_sizes:
        DeepSS_conv = DeepSS_input
        for i in range(0,nb_layers):
            DeepSS_conv = _conv_bn_relu1D(nb_filter=nb_filters, nb_row=fsz, subsample=1,use_bias=use_bias)(DeepSS_conv)
        DeepSS_conv = _conv_bn_softmax1D(nb_filter=1, nb_row=fsz, subsample=1,use_bias=use_bias,name='local_start')(DeepSS_conv)
        #DeepSS_conv = remove_1d_padding(ktop=ktop_node)(DeepSS_conv) ## remove the padding rows because they don't have targets
        #no need here, because if target is 0, the cross-entropy is zero, error will be not passed
        
        DeepSS_convs.append(DeepSS_conv)
    
    if len(filter_sizes)>1:
        DeepSS_out = Merge(mode='average')(DeepSS_convs)
    else:
        DeepSS_out = DeepSS_convs[0]  
    
    DeepSS_CNN = Model(input=[DeepSS_input], output=DeepSS_out)
    DeepSS_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    
    return DeepSS_CNN
