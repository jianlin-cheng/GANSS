# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:04 2017

@author: Jie Hou
"""
# python test_discriminator_variant1D.py
import sys
import os
from shutil import copyfile
GLOBAL_PATH='/storage/htc/bdm/jh7x3/GANSS/';
sys.path.insert(0, GLOBAL_PATH+'/lib/')
from keras.optimizers import Adam
from Model_construct import DeepCov_SS_with_paras
nb_filters = 5
nb_layers = 2
win_array = [10]
feature_num = 21
latent_size=100
AA_win=15
n_class=3
generator = DeepCov_SS_with_paras(win_array,feature_num,True,'sigmoid',nb_filters,nb_layers,'nadam')

print("\n\nInitializing the model");
generator.summary() 
