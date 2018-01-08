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

from Data_loading import load_train_test_data_for_gan
from Model_training import DeepSS_1dconv_gan_train_win_filter_layer_opt


import sys
if len(sys.argv) != 10:
          print 'please input the right parameters'
          sys.exit(1)

nb_filters=int(sys.argv[1]) #21
nb_layers_generator=int(sys.argv[2]) #10
nb_layers_discriminator=int(sys.argv[3]) #10
filtsize=sys.argv[4] #6_10
out_epoch=int(sys.argv[5]) #100
batch_size=int(sys.argv[6]) #1000
AA_win=int(sys.argv[7]) #15
feature_dir = sys.argv[8]
outputdir = sys.argv[9]

#nb_layers= 10
#filtsize= '10'
#out_epoch= 100
#batch_size= 1000
#AA_win=15
#feature_dir = '/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win15_no_atch_aa'
#outputdir = '/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/Parameter_tunning_win15_no_atch_aa/test'


test_datafile=GLOBAL_PATH+'/GANSS_Datasets/adj_dncon-test.lst'
train_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_training.list'
val_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_validation.list'


CV_dir=outputdir+'/filters'+str(nb_filters)+'_layersGen'+str(nb_layers_generator)+'_layersDis'+str(nb_layers_discriminator)+'_batch'+str(batch_size)+'_ftsize'+str(filtsize)+'_AA_win'+str(AA_win);

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
data_all_dict = load_train_test_data_for_gan(train_datafile, feature_dir)
testdata_all_dict = load_train_test_data_for_gan(val_datafile, feature_dir)

start_time = time.time()
DeepSS_1dconv_gan_train_win_filter_layer_opt(data_all_dict,testdata_all_dict,train_datafile,test_datafile,val_datafile,CV_dir,AA_win,feature_dir,"deepss_1dconv_gan",out_epoch,batch_size,filetsize_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir)
print("--- %s seconds ---" % (time.time() - start_time))


