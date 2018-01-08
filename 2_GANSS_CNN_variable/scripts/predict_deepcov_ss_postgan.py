# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:04 2017

@author: Jie Hou
"""
import sys
import os
from shutil import copyfile
GLOBAL_PATH='/storage/htc/bdm/jh7x3/GANSS/';
sys.path.insert(0, GLOBAL_PATH+'/lib/')

from keras.models import load_model, Sequential
import os
import numpy as np
import time

from Model_evaluation import DeepSS_1dconv_varigan_evaluation

if len(sys.argv) != 5:
          print 'please input the right parameters'
          sys.exit(1)

discriminator_model = sys.argv[1] # /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/results//filters15_layers2_batch1000_ftsize10/model-train-discriminator-deepss_1dconv_gan.hdf5 
feature_dir = sys.argv[2] # /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa
outputdir = sys.argv[3] #/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/results//filters15_layers2_batch1000_ftsize10/
AA_win = int(sys.argv[4])

#feature_dir = '/storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa'
#outputdir = '/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/Parameter_tunning_win15_no_atch_aa/test/'
test_datafile=GLOBAL_PATH+'/GANSS_Datasets/adj_dncon-test.lst'
train_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_training.list'
val_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_validation.list'


lib_dir=GLOBAL_PATH+'/lib/'

DeepSS_1dconv_varigan_evaluation(train_datafile,test_datafile,val_datafile,AA_win,discriminator_model,outputdir,feature_dir,lib_dir,True)
