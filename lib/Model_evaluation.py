# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os
from Model_construct import build_generator,build_discriminator,DeepCov_SS_with_paras
from keras.models import model_from_json,load_model, Sequential
import numpy as np
import time
import shutil
import shlex, subprocess
from subprocess import Popen, PIPE

from collections import defaultdict
#import cPickle as pickle
import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)


def DeepSS_1dconv_gan_evaluation(train_list,test_list,val_list,AA_win,discriminator_model,CV_dir,feature_dir,lib_dir,postGAN=False):
    import numpy as np
    
    Trainlist_data_keys = dict()
    Trainlist_targets_keys = dict()
    sequence_file=open(train_list,'r').readlines() 
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
    
    Testlist_data_keys = dict()
    Testlist_targets_keys = dict()
    sequence_file=open(test_list,'r').readlines() 
    for i in xrange(len(sequence_file)):
        pdb_name = sequence_file[i].rstrip()
        #print "Loading ",pdb_name
        featurefile = feature_dir + '/' + pdb_name + '.fea'
        if not os.path.isfile(featurefile):
                    print "feature file not exists: ",featurefile, " pass!"
                    continue           
        
        featuredata = np.loadtxt(featurefile) #(169, 51)
        fea_len = featuredata.shape[0]
        test_labels_tmp = featuredata[:,0:3]#(169, 3)
        test_feature_tmp = featuredata[:,3:] #(169, 48)    
        if pdb_name in Testlist_data_keys:
            print "Duplicate pdb name %s in Test list " % pdb_name
        else:
            Testlist_data_keys[pdb_name]=test_feature_tmp.reshape(test_feature_tmp.shape[0],AA_win,test_feature_tmp.shape[1]/AA_win)
        
        if pdb_name in Testlist_targets_keys:
            print "Duplicate pdb name %s in Test list " % pdb_name
        else:
            Testlist_targets_keys[pdb_name]=test_labels_tmp
    
    Vallist_data_keys = dict()
    Vallist_targets_keys = dict()
    sequence_file=open(val_list,'r').readlines() 
    for i in xrange(len(sequence_file)):
        pdb_name = sequence_file[i].rstrip()
        #print "Loading ",pdb_name
        featurefile = feature_dir + '/' + pdb_name + '.fea'
        if not os.path.isfile(featurefile):
                    print "feature file not exists: ",featurefile, " pass!"
                    continue           
        
        featuredata = np.loadtxt(featurefile) #(169, 51)
        fea_len = featuredata.shape[0]
        val_labels_tmp = featuredata[:,0:3]#(169, 3)
        val_feature_tmp = featuredata[:,3:] #(169, 48)  
        if pdb_name in Vallist_data_keys:
            print "Duplicate pdb name %s in Val list " % pdb_name
        else:
            Vallist_data_keys[pdb_name]=val_feature_tmp.reshape(val_feature_tmp.shape[0],AA_win,val_feature_tmp.shape[1]/AA_win)
        
        if pdb_name in Vallist_targets_keys:
            print "Duplicate pdb name %s in Val list " % pdb_name
        else:
            Vallist_targets_keys[pdb_name]=val_labels_tmp
    
    if os.path.exists(discriminator_model):
        print "######## Loading existing discriminator model ",discriminator_model;
        discriminator=Sequential()
        discriminator=load_model(discriminator_model) 
        
    else:    
        raise Exception("Failed to find model &s " % discriminator_model)
      
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    ### need evaluate the discriminator performance on classification
    sequence_file=open(test_list,'r').readlines() 
    predir = CV_dir + '/test_prediction/'
    chkdirs(predir)
    dnssdir = CV_dir + '/test_prediction_dnss/'
    chkdirs(dnssdir)
    eva_dir = CV_dir + '/test_prediction_q3_sov_log_loss/'
    chkdirs(eva_dir)
    
    test_acc=0.0;
    acc_num=0;
    for i in xrange(len(sequence_file)):
        pdb_name = sequence_file[i].rstrip()
        test_featuredata_all=Testlist_data_keys[pdb_name]
        test_targets=Testlist_targets_keys[pdb_name]
        result= discriminator.predict([test_featuredata_all])
        
        if postGAN:
            predict_val=result
        else:
            predict_val=result[1]
        #print "predict_val: ",predict_val.shape
        
        targsize=3
        predict_val= predict_val.reshape(predict_val.shape[0],predict_val.shape[1])
        max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
        #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
        preds = 1 * (predict_val > max_vals - .0001)
        preds_convert = np.argmax(preds, axis=1)
        test_labels_convert = np.argmax(test_targets, axis=1)
        
        test_acc +=float(sum(test_labels_convert == preds_convert))/len(test_labels_convert)
        acc_num += 1
        predfile = predir + pdb_name + ".pred";
        probfile = predir + pdb_name + ".prob";
        np.savetxt(predfile, preds, fmt="%d")
        np.savetxt(probfile, predict_val, fmt="%.6f")                        
        del test_featuredata_all
        del test_targets
    
    test_acc /= acc_num 
    
    
    args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-eva"
    print "Running "+ args_str
    args = shlex.split(args_str)
    pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
    
    scorefile=dnssdir+'/test_list-eva.score'
    
    found = 0
    while (found == 0):
        #print "Checking file ",scorefile
        time.sleep(10) 
        if os.path.exists(scorefile):
          found = 1
    
    shutil.copy2(scorefile, eva_dir)
    print "Score saved to file ",eva_dir
    ## clean for next iteration
    shutil.rmtree(predir)
    shutil.rmtree(dnssdir)
    
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
    for i in xrange(len(sequence_file)):
        pdb_name = sequence_file[i].rstrip()
        train_featuredata_all=Trainlist_data_keys[pdb_name]
        train_targets=Trainlist_targets_keys[pdb_name]
        
        result= discriminator.predict([train_featuredata_all])
        
        if postGAN:
            predict_val=result
        else:
            predict_val=result[1]
        #print "predict_val: ",predict_val.shape
          
        targsize=3
        predict_val= predict_val.reshape(predict_val.shape[0],predict_val.shape[1])
        max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
        #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
        preds = 1 * (predict_val > max_vals - .0001)
        preds_convert = np.argmax(preds, axis=1)
        train_targets_convert = np.argmax(train_targets, axis=1)
        
        train_acc +=float(sum(train_targets_convert == preds_convert))/len(train_targets_convert)
        acc_num += 1
        predfile = predir + pdb_name + ".pred";
        probfile = predir + pdb_name + ".prob";
        np.savetxt(predfile, preds, fmt="%d")
        np.savetxt(probfile, predict_val, fmt="%.6f")                        
        del train_featuredata_all
        del train_targets
    
    train_acc /= acc_num 
    
    args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-eva"
    print "Running "+ args_str
    args = shlex.split(args_str)
    pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
    
    scorefile=dnssdir+'/train_list-eva.score'
    
    found = 0
    while (found == 0):
        #print "Checking file ",scorefile
        time.sleep(20) 
        if os.path.exists(scorefile):
          found = 1
    
    shutil.copy2(scorefile, eva_dir)
    print "Score saved to file ",eva_dir
    ## clean for next iteration
    shutil.rmtree(predir)
    shutil.rmtree(dnssdir)

    
    
    ##### running validation
    sequence_file=open(val_list,'r').readlines() 
    predir = CV_dir + '/val_prediction/'
    chkdirs(predir)
    dnssdir = CV_dir + '/val_prediction_dnss/'
    chkdirs(dnssdir)
    eva_dir = CV_dir + '/val_prediction_q3_sov_log_loss/'
    chkdirs(eva_dir)
    
    val_acc=0.0;
    acc_num=0;
    for i in xrange(len(sequence_file)):
        pdb_name = sequence_file[i].rstrip()
        val_featuredata_all=Vallist_data_keys[pdb_name]
        val_targets=Vallist_targets_keys[pdb_name]
        
        result= discriminator.predict([val_featuredata_all])
        
        if postGAN:
            predict_val=result
        else:
            predict_val=result[1]
        #print "predict_val: ",predict_val.shape
          
        targsize=3
        predict_val= predict_val.reshape(predict_val.shape[0],predict_val.shape[1])
        max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
        #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
        preds = 1 * (predict_val > max_vals - .0001)
        preds_convert = np.argmax(preds, axis=1)
        val_targets_convert = np.argmax(val_targets, axis=1)
        
        val_acc +=float(sum(val_targets_convert == preds_convert))/len(train_targets_convert)
        acc_num += 1
        predfile = predir + pdb_name + ".pred";
        probfile = predir + pdb_name + ".prob";
        np.savetxt(predfile, preds, fmt="%d")
        np.savetxt(probfile, predict_val, fmt="%.6f")                        
        del val_featuredata_all
        del val_targets
    
    val_acc /= acc_num
    
    args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-eva"
    print "Running "+ args_str
    args = shlex.split(args_str)
    pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
    
    scorefile=dnssdir+'/val_list-eva.score'
    
    found = 0
    while (found == 0):
        #print "Checking file ",scorefile
        time.sleep(15) 
        if os.path.exists(scorefile):
          found = 1
    
    shutil.copy2(scorefile, eva_dir)
    print "Score saved to file ",eva_dir
    ## clean for next iteration
    shutil.rmtree(predir)
    shutil.rmtree(dnssdir)
    
