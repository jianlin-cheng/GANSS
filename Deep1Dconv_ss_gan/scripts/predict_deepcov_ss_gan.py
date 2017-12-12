# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:04 2017

@author: Jie Hou
"""
import sys
import os
from shutil import copyfile
sys.path.append('/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/lib/')  

from keras.models import load_model, Sequential
import os
import numpy as np
import time

if len(sys.argv) != 8:
          print 'please input the right parameters'
          sys.exit(1)

nb_layers=int(sys.argv[1]) #10
filtsize=sys.argv[2] #6_10
out_epoch=int(sys.argv[3]) #100
batch_size=int(sys.argv[4]) #1000
AA_win=int(sys.argv[5]) #15
feature_dir = sys.argv[6]
outputdir = sys.argv[7]

nb_layers= 10
filtsize= '10'
out_epoch= 100
batch_size= 1000
AA_win=15
feature_dir = '/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win15_no_atch_aa'
outputdir = '/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/Parameter_tunning_win15_no_atch_aa/test/'


test_datafile='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/adj_dncon-test.lst'
train_datafile='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/train_test_data/dncov_training.list'
val_datafile='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/train_test_data/dncov_validation.list'


CV_dir=outputdir+'/'+'_layers'+str(nb_layers)+'_batch'+str(batch_size)+'_ftsize'+str(filtsize);

CV_dir='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/Parameter_tunning_win15_no_atch_aa/test/layers10_batch1000_ftsize10_test';
lib_dir='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss_gan/lib/'


#CV_dir='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/ACGAN_jie_version/test';
Discriminator_path=CV_dir+'/params_discriminator_model_epoch_0_deepss_1dconv_gan.hdf5'
Generator_path=CV_dir+'/params_generator_model_epoch_0_deepss_1dconv_gan.hdf5'
Discriminator=Sequential()
Generator=Sequential()
Discriminator=load_model(Discriminator_path)
Generator=load_model(Generator_path)



pdb_name = '3CRY-A'
featurefile = feature_dir + '/' + pdb_name + '.fea'
featuredata = np.loadtxt(featurefile) #(169, 51)
fea_len = featuredata.shape[0]
train_labels_tmp = featuredata[:,0:3]#(169, 3)  # no need convert, this is for evaluation
train_feature_tmp = featuredata[:,3:] #(169, 315)     
train_feature = train_feature_tmp.reshape(train_feature_tmp.shape[0],AA_win,train_feature_tmp.shape[1]/AA_win) #(169, 15, 21)


result= Discriminator.predict([train_feature])
predict_val=result[1]
predict_val_truth=result[0]
  
targsize=3
predict_val= predict_val.reshape(predict_val.shape[0],predict_val.shape[1])
max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
#print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
preds = 1 * (predict_val > max_vals - .0001)
preds_convert = np.argmax(preds, axis=1)


train_labels_convert = np.argmax(train_labels_tmp, axis=1)

train_acc +=float(sum(train_labels_convert == preds_convert))/len(train_labels_convert)

predfile = predir + pdb_name + ".pred";
probfile = predir + pdb_name + ".prob";
np.savetxt(predfile, preds, fmt="%d")
np.savetxt(probfile, predict_val, fmt="%.6f")                        
del train_featuredata_all
del train_targets

    


Trainlist_data_keys = dict()
Trainlist_targets_keys = dict()
sequence_file=open(train_datafile,'r').readlines() 
for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    #print "Loading ",pdb_name
    featurefile = feature_dir + '/' + pdb_name + '.fea'
    if not os.path.isfile(featurefile):
                print "feature file not exists: ",featurefile, " pass!"
                continue           
    
    featuredata = np.loadtxt(featurefile) #(169, 51)
    fea_len = featuredata.shape[0]
    train_labels_tmp = featuredata[:,0:3]#(169, 3)  # no need convert, this is for evaluation
    train_feature_tmp = featuredata[:,3:] #(169, 48)        
    if pdb_name in Trainlist_data_keys:
        print "Duplicate pdb name %s in Train list " % pdb_name
    else:
        Trainlist_data_keys[pdb_name]=train_feature_tmp.reshape(train_feature_tmp.shape[0],AA_win,train_feature_tmp.shape[1]/AA_win)
    
    if pdb_name in Trainlist_targets_keys:
        print "Duplicate pdb name %s in Train list " % pdb_name
    else:
        Trainlist_targets_keys[pdb_name]=train_labels_tmp





##### running training
sequence_file=open(train_list,'r').readlines() 
predir = CV_dir + '/train_prediction/'
chkdirs(predir)
dnssdir = CV_dir + '/train_prediction_dnss/'
chkdirs(dnssdir)
eva_dir = CV_dir + '/train_prediction_q3_sov_log_loss/'
chkdirs(eva_dir)

train_acc=0.0;
acc_num=0;
train_loss=0.0;
loss_num=0;
for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    train_featuredata_all=Trainlist_data_keys[pdb_name]
    train_targets=Trainlist_targets_keys[pdb_name]
    
    #score, accuracy = DeepSS_CNN.evaluate([train_featuredata_all], train_targets, batch_size=10, verbose=0)
    #train_acc += accuracy
    #acc_num += 1
    
    #train_loss += score
    #loss_num += 1
    
    
    result= Discriminator.predict([train_featuredata_all])
    predict_val=result[1]
    predict_val_truth=result[0]
      
    targsize=3
    predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
    max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
    #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
    preds = 1 * (predict_val > max_vals - .0001)
    predfile = predir + pdb_name + ".pred";
    probfile = predir + pdb_name + ".prob";
    np.savetxt(predfile, preds, fmt="%d")
    np.savetxt(probfile, predict_val, fmt="%.6f")                        
    del train_featuredata_all
    del train_targets
    
    train_acc /= acc_num 
    train_loss /= loss_num 


    #noise = np.random.uniform(-1, 1, (1, 100))
    #sample_labels=np.zeros((1,1))
    #sample_labels[0]=number1
    #fake_image=Generator.predict([noise,sample_labels])
    #fake_image0=fake_image.reshape(1,1,28,28)





    result=Discriminator.predict(image)
    result_class=result[1]
    result_truth=result[0]
    print('\n*******************************************************')
    print('This image is truth or false: %f',result_truth)
    print('\n*******************************************************')
    print('The result of Class is shown in 【result_class】')
    print('\n*******************************************************')
    time.sleep(2)
try:
    number1=int(input('\nPlease inter the input you want to create:'))
except ValueError:
    print('\n*******************************************************')
    print('This is not a number!\nPlease Re-run this code.')
    print('*******************************************************')
else:
    noise = np.random.uniform(-1, 1, (1, 100))
    sample_labels=np.zeros((1,1))
    sample_labels[0]=number1
    fake_image=Generator.predict([noise,sample_labels])
    fake_image0=fake_image.reshape(1,1,28,28)
    result0=Discriminator.predict(fake_image0)
    fake_class=result0[1]
    fake_truth=result0[0]
    print('\n*******************************************************')
    print('This fake_image is truth or false: %f',fake_truth)
    print('\n*******************************************************')
    print('\n*******************************************************')
    print('The result of this fake_image Class is shown in【fake_class】')
    print('\n*******************************************************')
    time.sleep(2)
    
    
        
