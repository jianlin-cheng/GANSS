# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""

GLOBAL_PATH='/storage/htc/bdm/jh7x3/GANSS/';
sys.path.insert(0, GLOBAL_PATH+'/lib/')
import os
from Custom_class import remove_1d_padding
from Model_construct import DeepCov_SS_with_paras
from keras.models import model_from_json
import numpy as np
import time
import shutil
import shlex, subprocess
from subprocess import Popen, PIPE

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)


import sys
if len(sys.argv) != 7:
          print 'please input the right parameters'
          sys.exit(1)

test_list=(sys.argv[1]) #15
tag=sys.argv[2]
CV_dir=(sys.argv[3]) #10
model_in=(sys.argv[4]) #10
model_weight_in=sys.argv[5]
feature_dir=sys.argv[6]


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
    test_labels = featuredata[:,0:3]#(169, 3)
    test_feature = featuredata[:,3:] #(169, 48)    
    if pdb_name in Testlist_data_keys:
        print "Duplicate pdb name %s in Test list " % pdb_name
    else:
        Testlist_data_keys[pdb_name]=test_feature.reshape(1,test_feature.shape[0],test_feature.shape[1])
    
    if pdb_name in Testlist_targets_keys:
        print "Duplicate pdb name %s in Test list " % pdb_name
    else:
        Testlist_targets_keys[pdb_name]=test_labels.reshape(1,test_labels.shape[0],test_labels.shape[1])



### Define the model 

if os.path.exists(model_in):
    print "######## Loading existing model ",model_in;
    # load json and create model
    json_file_model = open(model_in, 'r')
    loaded_model_json = json_file_model.read()
    json_file_model.close()
    
    print("######## Loaded model from disk")
    #DeepSS_CNN = model_from_json(loaded_model_json, custom_objects={'remove_1d_padding': remove_1d_padding}) 
    DeepSS_CNN = model_from_json(loaded_model_json)       
else:
    print "######## Failed to find model",model_in;
    sys.exit(1)

if os.path.exists(model_weight_in):
    print "######## Loading existing weights ",model_weight_in;
    DeepSS_CNN.load_weights(model_weight_in)
    DeepSS_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='nadam')
else:
    print "######## Failed to find weight",model_weight_in;
    sys.exit(1)
 



## start evaluate the dataset
sequence_file=open(test_list,'r').readlines() 
predir = CV_dir + '/test_prediction/'
chkdirs(predir)
dnssdir = CV_dir + '/test_prediction_dnss/'
chkdirs(dnssdir)
eva_dir = CV_dir + '/test_prediction_q3_sov_log_loss/'
chkdirs(eva_dir)

test_acc=0.0;
acc_num=0;
test_loss=0.0;
loss_num=0;
for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    test_featuredata_all=Testlist_data_keys[pdb_name]
    test_targets=Testlist_targets_keys[pdb_name]
    score, accuracy = DeepSS_CNN.evaluate([test_featuredata_all], test_targets, batch_size=10, verbose=0)
    
    test_acc += accuracy
    acc_num += 1
    
    test_loss += score
    loss_num += 1
    predict_val= DeepSS_CNN.predict([test_featuredata_all])
    targsize=3
    predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
    max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
    #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
    preds = 1 * (predict_val > max_vals - .0001)
    predfile = predir + pdb_name + ".pred";
    probfile = predir + pdb_name + ".prob";
    np.savetxt(predfile, preds, fmt="%d")
    np.savetxt(probfile, predict_val, fmt="%.6f")
    
    del test_featuredata_all
    del test_targets

test_acc /= acc_num 
test_loss /= loss_num 

args_str ="perl " + GLOBAL_PATH + "/lib/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag " + tag
#print "Running "+ args_str
args = shlex.split(args_str)
pipe = subprocess.Popen(args, stdin=subprocess.PIPE)

scorefile=dnssdir+'/'+tag+'.score'

found = 0
while (found == 0):
    print "Checking file ",scorefile
    time.sleep(10) 
    if os.path.exists(scorefile):
      found = 1

shutil.copy2(scorefile, eva_dir)
print "Score saved to file ",eva_dir
## clean for next iteration
#shutil.rmtree(predir)
#shutil.rmtree(dnssdir)
    

