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
from Model_training import DeepSS_1dconv_gan_variant_train_win_filter_layer_opt

if len(sys.argv) != 11:
          print 'please input the right parameters'
          sys.exit(1)

inter=int(sys.argv[1]) #15
nb_filters=int(sys.argv[2]) #21
nb_layers_generator=int(sys.argv[3]) #10
nb_layers_discriminator=int(sys.argv[4]) #10
filtsize=sys.argv[5] #6_10
out_epoch=int(sys.argv[6]) #100
batch_size=int(sys.argv[7]) #1000
AA_win=int(sys.argv[8]) #15
feature_dir = sys.argv[9]
outputdir = sys.argv[10]

#inter = 15
#nb_filters = 5
#nb_layers_generator= 2
#nb_layers_discriminator= 2
#filtsize= '10'
#out_epoch= 100
#batch_size= 1000
#AA_win=15
#feature_dir = '/storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win1_no_atch_aa/'
#outputdir = '/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan_variableLen/results/'


test_datafile=GLOBAL_PATH+'/GANSS_Datasets/adj_dncon-test.lst'
train_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_training.list'
val_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_validation.list'


CV_dir=outputdir+'/inter'+str(inter)+'_filters'+str(nb_filters)+'_layersGen'+str(nb_layers_generator)+'_layersDis'+str(nb_layers_discriminator)+'_batch'+str(batch_size)+'_ftsize'+str(filtsize);

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
DeepSS_1dconv_gan_variant_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_datafile,test_datafile,val_datafile,CV_dir,AA_win,feature_dir,"deepss_1dconv_varigan",out_epoch,inter, 5000 ,batch_size,filetsize_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir)
print("--- %s seconds ---" % (time.time() - start_time))
