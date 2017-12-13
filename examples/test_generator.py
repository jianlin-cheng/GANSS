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
from Model_construct import build_generator
nb_filters = 5
nb_layers = 2
win_array = [10]
fea_num = 21
latent_size=100
AA_win=15
n_class=3
generator = build_generator(latent_size,AA_win,nb_filters,nb_layers,win_array,fea_num,n_class)

print("\n\nInitializing the Generator");
generator.summary() 
