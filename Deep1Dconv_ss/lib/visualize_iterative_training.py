# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:04 2017

@author: Jie Hou
"""
import sys
sys.path.append('/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/')  

from Data_loading import load_train_test_data_padding_with_interval,load_train_test_data_padding_with_interval_withcontact
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training,DLS2F_train_withaa_efficient_complex,DLS2F_train_withaa_efficient_complex_withcontact,DLS2F_train_withaa_efficient_complex_withcontact2D, DLS2F_train_withaa_efficient_complex_win

## on iris
import sys
sys.path.append('/home/jh7x3/DLS2F/DLS2F_Project/Paper_data/Models/Family_level/1_Final_scripts_test_20170222_lewis_kmax30_for_visualize_training') 

from Data_loading import load_train_test_data_padding_with_interval
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training


 
######  family level. SCOP95_70 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win/'





 
######  family level. SCOP95 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95'





 
######  family level. SCOP95_70_40 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40'



 
######  family level. SCOP95_70_40_25 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25'



 
######  family level. SCOP95_70_40_25_15 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15'



######  family level. SCOP95_70 on lewis using ratio 70 20 10% for all iterative training 
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_range0-500/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_range0-500/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_range0-500/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_range0-500'



### SCOP95_70 ---done
print "Running interval500 for iterative\n\n"
data_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir, 500,'kmax30',400,train=True)
testdata_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir,500,'kmax30',400,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval500,testdata_all_dict_padding_interval500,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,500,check_list)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.44220
The top1_acc accuracy2 is 0.44220
The top5_acc accuracy is 0.69153
The top10_acc accuracy is 0.77755
The top15_acc accuracy is 0.82930
The top20_acc accuracy is 0.85148


import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval500,testdata_all_dict_padding_interval500,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,400)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))

print "Running interval50 only 1000 epoch on lewis\n\n"
data_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir, 500,'kmax30',400,train=True)
testdata_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir,500,'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval500,testdata_all_dict_padding_interval500,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-int500-only-1000epoch",200,5,50,check_list)
print("--- %s seconds ---" % (time.time() - start_time))




print "Running interval500 for iterative\n\n"
data_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir, 500,'kmax30',500,train=True)
testdata_all_dict_padding_interval500 = load_train_test_data_padding_with_interval(CV_dir,500,'kmax30',500,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval500,testdata_all_dict_padding_interval500,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,500,check_list)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))


## SCOP95_70 --- done
print "Running interval200 for iterative\n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',400,train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',400,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,200,check_list)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))




## SCOP95 --- done
print "Running interval200\n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',400,train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',400,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,400)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.32191
The top1_acc accuracy2 is 0.32191
The top5_acc accuracy is 0.57228
The top10_acc accuracy is 0.67362
The top15_acc accuracy is 0.74416
The top20_acc accuracy is 0.78887



## SCOP95_70_40 --- done
print "Running interval200\n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',400,train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',400,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,400)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.29326
The top1_acc accuracy2 is 0.29326
The top5_acc accuracy is 0.52746
The top10_acc accuracy is 0.61451
The top15_acc accuracy is 0.68497
The top20_acc accuracy is 0.73368



## SCOP95_70_40_25 --- done
print "Running interval200\n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',400,train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',400,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,400)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.17707
The top1_acc accuracy2 is 0.17707
The top5_acc accuracy is 0.41655
The top10_acc accuracy is 0.53701
The top15_acc accuracy is 0.61974
The top20_acc accuracy is 0.66473



print "Running interval200 for iterative\n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',500,train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',500,train=False)

import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,30,200,check_list)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))



import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,20)
print("--- Running interval200: %s seconds ---" % (time.time() - start_time))


print "Running interval200 only 1000 epoch on lewis \n\n"
data_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir, 200,'kmax30',train=True)
testdata_all_dict_padding_interval200 = load_train_test_data_padding_with_interval(CV_dir,200,'kmax30',train=False)




import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval200,testdata_all_dict_padding_interval200,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-int200-only-1000epoch",200,5,200,check_list)


## SCOP95_70 --- running
print "Running interval100 for iterative\n\n"
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir,100, 'kmax30',400,train=False)
import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,100,check_list)

print("--- %s seconds ---" % (time.time() - start_time))


## SCOP95 --- done
print "Running interval100\n\n"
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir,100, 'kmax30',400,train=False)
import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)
The test accuracy is 0.61351
The top1_acc accuracy2 is 0.61351
The top5_acc accuracy is 0.83358
The top10_acc accuracy is 0.89170
The top15_acc accuracy is 0.91952
The top20_acc accuracy is 0.93393
The val accuracy is 0.61351
The training accuracy is 0.59759



## SCOP95_70_40 --- done
print "Running interval100\n\n"
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir,100, 'kmax30',400,train=False)
import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)

print("--- %s seconds ---" % (time.time() -start_time))


## SCOP95_70_40_25 --- done
print "Running interval100\n\n"
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir,100, 'kmax30',400,train=False)
import time
start_time = time.time()
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)

print("--- %s seconds ---" % (time.time() -start_time))
Training finished, best testing acc =  0.410740203193
Training finished, best validation acc =  0.410740203193
Training finished, best top1 acc =  0.410740203193
Training finished, best top5 acc =  0.647314949202
Training finished, best top10 acc =  0.750362844702
Training finished, best top15 acc =  0.801161103048
Training finished, best top20 acc =  0.831640058055


print "Running interval100 only 1000 epoch on iris \n\n"
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir, 100, 'kmax30',train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval(CV_dir,100, 'kmax30',train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,5)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-int100-only-1000epoch",200,5,100,check_list)
# -> test: 0.09 train: 0.12267
print("--- %s seconds ---" % (time.time() - start_time))

## SCOP95_70  --- done
print "Running interval50\n\n"
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir,50, 'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,50,check_list)
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best training acc =  0.775266067156
Training finished, best testing acc =  0.694892473118
Training finished, best validation acc =  0.694892473118
Training finished, best top1 acc =  0.694892473118
Training finished, best top5 acc =  0.86626344086
Training finished, best top10 acc =  0.913306451613
Training finished, best top15 acc =  0.938172043011
Training finished, best top20 acc =  0.952284946237

## SCOP95 --- running
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir,50, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))

## SCOP95_70_40 --- done
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir,50, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))

The test accuracy is 0.49326
The top1_acc accuracy2 is 0.49326
The top5_acc accuracy is 0.71917
The top10_acc accuracy is 0.79171
The top15_acc accuracy is 0.84249
The top20_acc accuracy is 0.86632
The val accuracy is 0.49326
The training accuracy is 0.60780




## SCOP95_70_40_25 --- done
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir,50, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.35414
The top1_acc accuracy2 is 0.35414
The top5_acc accuracy is 0.60377
The top10_acc accuracy is 0.72569
The top15_acc accuracy is 0.78229
The top20_acc accuracy is 0.83745
The val accuracy is 0.35414
The training accuracy is 0.58850


print "Running interval50 only 1000 epoch on lewis\n\n"
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval(CV_dir,50, 'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-int50-only-1000epoch",200,5,50,check_list)
print("--- %s seconds ---" % (time.time() - start_time))



## SCOP95_70  --- running
print "Running interval30\n\n"
data_all_dict_padding_interval30 = load_train_test_data_padding_with_interval(CV_dir, 30, 'kmax30',400,train=True)
testdata_all_dict_padding_interval30 = load_train_test_data_padding_with_interval(CV_dir,30, 'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval30,testdata_all_dict_padding_interval30,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,30,check_list)
print("--- %s seconds ---" % (time.time() - start_time))


## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,3,15,check_list)
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best training acc =  0.874845309793
Training finished, best testing acc =  0.752688172043
Training finished, best validation acc =  0.752688172043
Training finished, best top1 acc =  0.752688172043
Training finished, best top5 acc =  0.91935483871
Training finished, best top10 acc =  0.944220430108
Training finished, best top15 acc =  0.957661290323
Training finished, best top20 acc =  0.967069892473


## SCOP95 --- running
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",1,1,400)
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best testing acc =  0.806259314456
Training finished, best validation acc =  0.806259314456
Training finished, best top1 acc =  0.806259314456
Training finished, best top5 acc =  0.932935916542
Training finished, best top10 acc =  0.955787382017
Training finished, best top15 acc =  0.967213114754
Training finished, best top20 acc =  0.972677595628


## SCOP95 increase to 1500 --- running
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",10,2,1150)
Training finished, best testing acc =  0.781917536016
Training finished, best validation acc =  0.781917536016
Training finished, best top1 acc =  0.781917536016
Training finished, best top5 acc =  0.931942374565
Training finished, best top10 acc =  0.960755091903
Training finished, best top15 acc =  0.968206656731
Training finished, best top20 acc =  0.973174366617






## SCOP95_70_40 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",15,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.50052
The top1_acc accuracy2 is 0.50052
The top5_acc accuracy is 0.75959
The top10_acc accuracy is 0.83005
The top15_acc accuracy is 0.87772
The top20_acc accuracy is 0.89948
The val accuracy is 0.50052
The training accuracy is 0.65314


## SCOP95_70_40_25 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",15,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best testing acc =  0.486211901306
Training finished, best validation acc =  0.486211901306
Training finished, best top1 acc =  0.486211901306
Training finished, best top5 acc =  0.715529753266
Training finished, best top10 acc =  0.789550072569
Training finished, best top15 acc =  0.846153846154
Training finished, best top20 acc =  0.869375907112



## SCOP95_70_40_25_15 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",15,3,400)
print("--- %s seconds ---" % (time.time() - start_time))

print "Running interval15\n\n"
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",10,3,15,check_list)
print("--- %s seconds ---" % (time.time() - start_time))
# interval15

# now loading model and predict on whole dataset with few epoch

print "Running interval15\n\n"
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1500,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1500,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",2,3,15,check_list)



print "Running interval15  using pretrained weight on ratio 7 2 1 SCOP9570\n\n"
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-pretrained",30,3)
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best training acc =  0.884417129262
Training finished, best testing acc =  0.713331105914
Training finished, best validation acc =  0.863636363636
Training finished, best top1 acc =  0.713331105914
Training finished, best top5 acc =  0.896424991647
Training finished, best top10 acc =  0.934847978617
Training finished, best top15 acc =  0.953224189776
Training finished, best top20 acc =  0.961911125961



print "Running interval15  only 1000 epoch on lewis \n\n"
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-int15-only-1000epoch",200,5,15,check_list)
print("--- %s seconds ---" % (time.time() - start_time))

# SCOP95_70 --- running
print "Running interval10\n\n"
data_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir, 10, 'kmax30',400,train=True)
testdata_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir,10,'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex_record_iterative_training(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",20,3,10,check_list)
print("--- %s seconds ---" % (time.time() - start_time))


# SCOP95 --- done
print "Running interval10\n\n"
data_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir, 10, 'kmax30',400,train=True)
testdata_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir,10,'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))



# SCOP95_70_40_25 --- running
print "Running interval10\n\n"
data_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir, 10, 'kmax30',400,train=True)
testdata_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir,10,'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.48331
The top1_acc accuracy2 is 0.48331
The top5_acc accuracy is 0.71408
The top10_acc accuracy is 0.80697
The top15_acc accuracy is 0.83599
The top20_acc accuracy is 0.86357



# SCOP95_70_40 --- running
print "Running interval10\n\n"
data_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir, 10, 'kmax30',400,train=True)
testdata_all_dict_padding_interval10 = load_train_test_data_padding_with_interval(CV_dir,10,'kmax30',400,train=False)
import time
start_time = time.time()
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.63212
The top1_acc accuracy2 is 0.63212
The top5_acc accuracy is 0.83109
The top10_acc accuracy is 0.90259
The top15_acc accuracy is 0.92332
The top20_acc accuracy is 0.94197
Saved best weight to disk
The val accuracy is 0.63212
The training accuracy is 0.88405


print "Running interval2\n\n"
data_all_dict_padding_interval2 = load_train_test_data_padding_with_interval(CV_dir,2, 'kmax30',train=True)
testdata_all_dict_padding_interval2 = load_train_test_data_padding_with_interval(CV_dir,2, 'kmax30',train=False)
#DLS2F_train_withaa_efficient(data_all_dict_padding_interval5,testdata_all_dict_padding_interval5,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",30,2)
DLS2F_train_withaa_efficient_complex(data_all_dict_padding_interval5,testdata_all_dict_padding_interval5,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30",30,2) 
#DLS2F_train_withaa_efficient_for_recall(data_all_dict_padding_interval5,testdata_all_dict_padding_interval5,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",15,2,2,check_list)
# -> test: 0.78 train: 0.92








##############################  run with contact 
 
import sys
sys.path.append('/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/')  

from Data_loading import load_train_test_data_padding_with_interval,load_train_test_data_padding_with_interval_withcontact
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training,DLS2F_train_withaa_efficient_complex,DLS2F_train_withaa_efficient_complex_withcontact
######  family level. SCOP95 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact'



test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2'



## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval_withcontact(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval_withcontact(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))



data_all_dict_padding_interval10 = load_train_test_data_padding_with_interval_withcontact(CV_dir, 10, 'kmax30',400,train=True)
testdata_all_dict_padding_interval10 = load_train_test_data_padding_with_interval_withcontact(CV_dir,10, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact(data_all_dict_padding_interval10,testdata_all_dict_padding_interval10,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))




import time
## SCOP95_70 --- done
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval_withcontact(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval_withcontact(CV_dir,100, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))
The test accuracy is 0.66518
The top1_acc accuracy2 is 0.66518
The top5_acc accuracy is 0.86538
The top10_acc accuracy is 0.91654
The top15_acc accuracy is 0.93741
The top20_acc accuracy is 0.95380
The val accuracy is 0.66518
The training accuracy is 0.67025




import time
## SCOP95_70 --- done
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval_withcontact(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval_withcontact(CV_dir,50, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",20,3,400)
print("--- %s seconds ---" % (time.time() - start_time))








##############################  run with contact 2D
 
import sys
sys.path.append('/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/')  


from Data_loading import load_train_test_data_padding_with_interval,load_train_test_data_padding_with_interval_withcontact,load_train_test_data_padding_with_interval_withcontact2D
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training,DLS2F_train_withaa_efficient_complex,DLS2F_train_withaa_efficient_complex_withcontact,DLS2F_train_withaa_efficient_complex_withcontact2D
######  family level. SCOP95 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2D/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2D/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2D/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_withcontact2D'




import time
## SCOP95_70 --- done
data_all_dict_padding_interval100 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir, 100, 'kmax30',400,train=True)
testdata_all_dict_padding_interval100 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir,100, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact2D(data_all_dict_padding_interval100,testdata_all_dict_padding_interval100,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))



import time
## SCOP95_70 --- done
data_all_dict_padding_interval50 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir, 50, 'kmax30',400,train=True)
testdata_all_dict_padding_interval50 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir,50, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact2D(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))




import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir, 15, 'kmax30',400,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval_withcontact2D(CV_dir,15, 'kmax30',400,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_withcontact2D(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-withcontact",30,3,400)
print("--- %s seconds ---" % (time.time() - start_time))







##################### 20170317  try different window size 

import sys
sys.path.append('/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/')  

from Data_loading import load_train_test_data_padding_with_interval,load_train_test_data_padding_with_interval_withcontact
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training,DLS2F_train_withaa_efficient_complex,DLS2F_train_withaa_efficient_complex_withcontact,DLS2F_train_withaa_efficient_complex_withcontact2D, DLS2F_train_withaa_efficient_complex_win

## on iris
import sys
sys.path.append('/home/jh7x3/DLS2F/DLS2F_Project/Paper_data/Models/Family_level/1_Final_scripts_test_20170222_lewis_kmax30_for_visualize_training') 

from Data_loading import load_train_test_data_padding_with_interval
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training


 
######  family level. SCOP95_70 on lewis
check_list = ['d1eb7a1','d1w0ma_','d1eysl_','d1myta_','d1r2ja2','d1k3ka_','d1lvaa4','d1wu4a1','d1pc2a_','d1qnzh_','d1qasa2','d1nlsa_','d1iz0a1','d1bqya','d1wfha_','d2nllb_','d1tg7a2','d2vpaa1','d1e5wa2','d2gm6a1']

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win/'




import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_nobias-relu",200,3,1150,[6],False,'relu')
print("--- %s seconds ---" % (time.time() - start_time))


import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_nobias-relu",100,3,1150,[6,10],False,'relu')
print("--- %s seconds ---" % (time.time() - start_time))




import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_nobias-sigmoid",200,3,1150,[6,10],False,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))

Training finished, best testing acc =  0.789865871833
Training finished, best validation acc =  0.789865871833
Training finished, best top1 acc =  0.789865871833
Training finished, best top5 acc =  0.921510183805
Training finished, best top10 acc =  0.953800298063
Training finished, best top15 acc =  0.964729259811
Training finished, best top20 acc =  0.973671137606


import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",200,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))

The test accuracy is 0.80378
The top1_acc accuracy2 is 0.80378
The top5_acc accuracy is 0.93691
The top10_acc accuracy is 0.96225
The top15_acc accuracy is 0.97317
The top20_acc accuracy is 0.97963




import time
## SCOP95_70 --- done
data_all_dict_padding_interval5 = load_train_test_data_padding_with_interval(CV_dir, 5, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval5 = load_train_test_data_padding_with_interval(CV_dir,5, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval5,testdata_all_dict_padding_interval5,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",20,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))



##############################################

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_win/'


data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_nobias-sigmoid",50,3,1150,[6,10],False,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))




import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",100,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))
Training finished, best testing acc =  0.794833581719
Training finished, best validation acc =  0.794833581719
Training finished, best top1 acc =  0.794833581719
Training finished, best top5 acc =  0.929955290611
Training finished, best top10 acc =  0.958768007948
Training finished, best top15 acc =  0.970193740686
Training finished, best top20 acc =  0.976651763537




from model_evaluation import evaluate_model_performance
test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'

model_in='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win/model_ResCNN_pssm-train-iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid.json'

model_weight='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win/model_ResCNN_pssm-train-weight-iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid_Te80.h5'


evaluate_model_performance(test_datafile,model_in,model_weight)





##############################################

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_win/'



import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",200,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))




##############################################

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_win/'



import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",200,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))





##############################################

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15_win//Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_win//validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion//Training_data/Ratio_8_2/SCOP95_70_40_25_15_win/'
import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",50,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))







##### using all SCOP95 data to train the model 

import sys
sys.path.append('/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/')  

from Data_loading import load_train_test_data_padding_with_interval,load_train_test_data_padding_with_interval_withcontact
from Model_training import DLS2F_train_withaa_efficient_complex_record_iterative_training,DLS2F_train_withaa_efficient_complex,DLS2F_train_withaa_efficient_complex_withcontact,DLS2F_train_withaa_efficient_complex_withcontact2D, DLS2F_train_withaa_efficient_complex_win

test_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/Training_data/Ratio_8_2/SCOP95_win_train_all_data_for_website/Testdata.list'
train_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/Training_data/Ratio_8_2/SCOP95_win_train_all_data_for_website/Traindata.list'
val_datafile='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/Training_data/Ratio_8_2/SCOP95_win_train_all_data_for_website/validation.list'
CV_dir='/data/jh7x3/DLS2F/DLS2F_Project/cactus/PDB_SCOP95_SEQ/New_training_strategy_20170224/4_Final_scripts_test_20170309_lewis_kmax30_for_model_construction_finalversion/Training_data/Ratio_8_2/SCOP95_win_train_all_data_for_website/'
import time
## SCOP95_70 --- done
data_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir, 15, 'kmax30',1150,train=True)
testdata_all_dict_padding_interval15 = load_train_test_data_padding_with_interval(CV_dir,15, 'kmax30',1150,train=False)
start_time = time.time()
#DLS2F_train_withaaa_efficient(data_all_dict_padding_interval50,testdata_all_dict_padding_interval50,train_datafile,test_datafile,CV_dir,set_id,"5-fold-padding-withaa-interval100",50,3)
DLS2F_train_withaa_efficient_complex_win(data_all_dict_padding_interval15,testdata_all_dict_padding_interval15,train_datafile,test_datafile,val_datafile,CV_dir,"iterative-padding-withaa-complex-kmax30-win6_10_bias-sigmoid",50,3,1150,[6,10],True,'sigmoid')
print("--- %s seconds ---" % (time.time() - start_time))





