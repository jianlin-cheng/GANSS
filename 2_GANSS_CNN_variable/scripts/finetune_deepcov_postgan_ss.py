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
from Model_training import DeepSS_1dconv_fixed_window_train_win_filter_layer_opt,DeepSS_1dcon_finetune_variant_train_win_filter_layer_opt


import sys
if len(sys.argv) != 11:
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
postGAN_model = sys.argv[10] #/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/results//filters15_layers2_batch1000_ftsize10/model-train-discriminator-deepss_1dconv_postgan.hdf5

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


CV_dir=outputdir;

lib_dir=GLOBAL_PATH+'/lib/'


filetsize_array = map(int,filtsize.split("_"))

if not os.path.exists(CV_dir):
    os.makedirs(CV_dir)



import time
data_all_dict_padding = load_train_test_data_padding_with_interval(train_datafile, feature_dir, 15,5000, 'deepss_1dconv')
testdata_all_dict_padding = load_train_test_data_padding_with_interval(val_datafile, feature_dir, 15,5000, 'deepss_1dconv')

start_time = time.time()
DeepSS_1dcon_finetune_variant_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_datafile,test_datafile,val_datafile,CV_dir,AA_win,feature_dir,"deepss_1dconv_postgan_finetune",out_epoch,15, 5000,batch_size,filetsize_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,postGAN = postGAN_model)
print("--- %s seconds ---" % (time.time() - start_time))


