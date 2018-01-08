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

from Data_loading import load_train_test_data_padding_with_interval
from Model_training import DeepSS_1dconv_train_win_filter_layer_opt


import sys
if len(sys.argv) != 10:
          print 'please input the right parameters'
          sys.exit(1)

inter=int(sys.argv[1]) #15
nb_filters=int(sys.argv[2]) #10
nb_layers=int(sys.argv[3]) #10
opt=sys.argv[4] #nadam
filtsize=sys.argv[5] #6_10
out_epoch=int(sys.argv[6]) #100
in_epoch=int(sys.argv[7]) #3
feature_dir = sys.argv[8]
outputdir = sys.argv[9]


test_datafile=GLOBAL_PATH+'/GANSS_Datasets/adj_dncon-test.lst'
train_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_training.list'
val_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_validation.list'


CV_dir=outputdir+'/filter'+str(nb_filters)+'_layers'+str(nb_layers)+'_inter'+str(inter)+'_opt'+str(opt)+'_ftsize'+str(filtsize);

lib_dir=GLOBAL_PATH+'/lib/'
"""
modelfile = CV_dir+'/model-train-deepss_1dconv.json'
weightfile = CV_dir+'/model-train-weight-deepss_1dconv.h5'
weightfile_best = CV_dir+'/model-train-weight-deepss_1dconv-best-val.h5'


if os.path.exists(modelfile):
  cmd1='rm  '+ modelfile
  print "Running ", cmd1,"\n\n"
  os.system(cmd1)
  

if os.path.exists(weightfile_best):
  cmd1='cp  '+ weightfile_best + '  ' + weightfile
  print "Running ", cmd1,"\n\n"
  os.system(cmd1)
"""

filetsize_array = map(int,filtsize.split("_"))

if not os.path.exists(CV_dir):
    os.makedirs(CV_dir)



import time
data_all_dict_padding = load_train_test_data_padding_with_interval(train_datafile, feature_dir, inter,5000, 'deepss_1dconv')
testdata_all_dict_padding = load_train_test_data_padding_with_interval(val_datafile, feature_dir, inter,5000, 'deepss_1dconv')

start_time = time.time()
DeepSS_1dconv_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_datafile,test_datafile,val_datafile,CV_dir, feature_dir,"deepss_1dconv",out_epoch,in_epoch,inter,5000,filetsize_array,True,'sigmoid',nb_filters,nb_layers,opt,lib_dir)

print("--- %s seconds ---" % (time.time() - start_time))
