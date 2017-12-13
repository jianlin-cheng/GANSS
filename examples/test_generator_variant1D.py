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
from Model_construct import build_generator_variant1D
nb_layers = 2
win_array = [10]
fea_num = 21
n_class =3
nb_filters_generator = fea_num
latent_size=100
generator_variant1D = build_generator_variant1D(latent_size,nb_filters_generator,nb_layers,win_array,fea_num,n_class)

print("\n\nInitializing the variant1D Discriminator");
generator_variant1D.summary()
