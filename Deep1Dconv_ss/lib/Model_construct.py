# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2017

@author: Jie Hou
"""

from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization


from Custom_class import remove_1d_padding



# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(nb_filter, nb_row, subsample,name,use_bias=True):
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='relu', border_mode="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(mode=0, axis=2,name="%s_nor" % name)(conv)
        return Dense(output_dim=3, init="he_normal",name="%s_softmax" % name, activation="softmax")(norm)
    
    return f



# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(nb_filter, nb_row, subsample,use_bias=True):
    def f(input):
        conv = Convolution1D(nb_filter=nb_filter, filter_length=nb_row, subsample_length=subsample,bias=use_bias,
                             init="he_normal", activation='relu', border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=2)(conv)
        return Activation("relu")(norm)
    
    return f


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

