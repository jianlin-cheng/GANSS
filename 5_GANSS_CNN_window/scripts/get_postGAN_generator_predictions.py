# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:04 2017

@author: Jie Hou
"""
import sys
import os
import numpy as np
from shutil import copyfile
GLOBAL_PATH='/storage/htc/bdm/jh7x3/GANSS/';
sys.path.insert(0, GLOBAL_PATH+'/lib/')

from keras.models import Sequential, Model,load_model
from Model_construct import build_discriminator_postGAN
from keras.optimizers import Adam

import sys
if len(sys.argv) != 5:
          print 'please input the right parameters'
          sys.exit(1)

model_in=sys.argv[1] #21
feature_dir=sys.argv[2]
AA_win=int(sys.argv[3])
outputdir=sys.argv[4]

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

test_datafile=GLOBAL_PATH+'/GANSS_Datasets/adj_dncon-test.lst'
train_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_training.list'
val_datafile=GLOBAL_PATH+'/GANSS_Datasets/dncov_validation.list'

import time



if os.path.exists(model_in):
    print "######## Loading existing generator model ",model_in;
    generator=Sequential()
    generator=load_model(model_in)
else:    
    raise Exception("Failed to find model (%s) " % model_in)


print "\n\n#### Summary of generator: ";
print(generator.summary())

adam_lr = 0.00005
adam_beta_1 = 0.5
generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
	
latent_size = 100


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
    train_labels_convert = np.argmax(train_labels_tmp, axis=1) ## need convert to number for gan
    train_feature_tmp = featuredata[:,3:] #(169, 48)        
    if pdb_name in Trainlist_data_keys:
        print "Duplicate pdb name %s in Train list " % pdb_name
    else:
        Trainlist_data_keys[pdb_name]=train_feature_tmp.reshape(train_feature_tmp.shape[0],AA_win,train_feature_tmp.shape[1]/AA_win)
    
    if pdb_name in Trainlist_targets_keys:
        print "Duplicate pdb name %s in Train list " % pdb_name
    else:
        Trainlist_targets_keys[pdb_name]=train_labels_convert

print "!!! Training data loaded\n\n"

Testlist_data_keys = dict()
Testlist_targets_keys = dict()
sequence_file=open(test_datafile,'r').readlines() 
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
    test_labels_tmp_convert = np.argmax(test_labels_tmp, axis=1) ## need convert to number for gan
    test_feature_tmp = featuredata[:,3:] #(169, 48)    
    if pdb_name in Testlist_data_keys:
        print "Duplicate pdb name %s in Test list " % pdb_name
    else:
        Testlist_data_keys[pdb_name]=test_feature_tmp.reshape(test_feature_tmp.shape[0],AA_win,test_feature_tmp.shape[1]/AA_win)
    
    if pdb_name in Testlist_targets_keys:
        print "Duplicate pdb name %s in Test list " % pdb_name
    else:
        Testlist_targets_keys[pdb_name]=test_labels_tmp_convert

print "!!! Testing data loaded\n\n"


Vallist_data_keys = dict()
Vallist_targets_keys = dict()
sequence_file=open(val_datafile,'r').readlines() 
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
    val_labels_tmp_convert = np.argmax(val_labels_tmp, axis=1) ## need convert to number for gan
    val_feature_tmp = featuredata[:,3:] #(169, 48)  
    if pdb_name in Vallist_data_keys:
        print "Duplicate pdb name %s in Val list " % pdb_name
    else:
        Vallist_data_keys[pdb_name]=val_feature_tmp.reshape(val_feature_tmp.shape[0],AA_win,val_feature_tmp.shape[1]/AA_win)
    
    if pdb_name in Vallist_targets_keys:
        print "Duplicate pdb name %s in Val list " % pdb_name
    else:
        Vallist_targets_keys[pdb_name]=val_labels_tmp_convert


print "!!! Validation data loaded\n\n"

##### running training
sequence_file=open(train_datafile,'r').readlines() 

train_data_history_out = "%s/train.reconstruction_history" % (outputdir)
chkdirs(train_data_history_out)     
with open(train_data_history_out, "w") as myfile:
  myfile.write("PDB_name\tReconstructionError\n")

for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    train_featuredata_all=Trainlist_data_keys[pdb_name]
    train_targets=Trainlist_targets_keys[pdb_name]
    
    nb_test = len(train_targets)
    noise = np.random.uniform(-1, 1, (nb_test, latent_size))
    reconstruct_samples = generator.predict([noise, train_targets.reshape((-1, 1))], verbose=0)
    reconstruction_test_error = ((train_featuredata_all - reconstruct_samples) ** 2).mean()
    
    reconstruct_pssm = reconstruct_samples[:,(AA_win-1)/2,0:20] 
    original_pssm = train_featuredata_all[:,(AA_win-1)/2,0:20] 
    reconstruct_pssm_outfile = outputdir + '/'+pdb_name+'_gan.pssm_fea'
    np.savetxt(reconstruct_pssm_outfile, reconstruct_pssm, fmt='%.4f') 
    
    original_pssm_outfile = outputdir + '/'+pdb_name+'_orig.pssm_fea'
    np.savetxt(original_pssm_outfile, original_pssm, fmt='%.4f') 
    
    reconstruct_samples_reshape = reconstruct_samples.reshape(reconstruct_samples.shape[0],reconstruct_samples.shape[1]*reconstruct_samples.shape[2])
    print "Generating gan feature for: ",pdb_name, " -----> Reconstruction error: ",reconstruction_test_error
    outfile = outputdir + '/'+pdb_name+'_gan.fea'
    np.savetxt(outfile, reconstruct_samples_reshape, fmt='%.4f')
    
    content = "%s\t%.4f\n" % (pdb_name,reconstruction_test_error)
    with open(train_data_history_out, "a") as myfile:
        myfile.write(content)
                


##### running testing
sequence_file=open(test_datafile,'r').readlines() 

test_data_history_out = "%s/test.reconstruction_history" % (outputdir)
chkdirs(test_data_history_out)     
with open(test_data_history_out, "w") as myfile:
  myfile.write("PDB_name\tReconstructionError\n")

for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    test_featuredata_all=Testlist_data_keys[pdb_name]
    test_targets=Testlist_targets_keys[pdb_name]
    
    nb_test = len(test_targets)
    noise = np.random.uniform(-1, 1, (nb_test, latent_size))
    reconstruct_samples = generator.predict([noise, test_targets.reshape((-1, 1))], verbose=0)
    reconstruction_test_error = ((test_featuredata_all - reconstruct_samples) ** 2).mean()
    
    reconstruct_pssm = reconstruct_samples[:,(AA_win-1)/2,0:20] 
    original_pssm = test_featuredata_all[:,(AA_win-1)/2,0:20] 
    reconstruct_pssm_outfile = outputdir + '/'+pdb_name+'_gan.pssm_fea'
    np.savetxt(reconstruct_pssm_outfile, reconstruct_pssm, fmt='%.4f') 
    
    original_pssm_outfile = outputdir + '/'+pdb_name+'_orig.pssm_fea'
    np.savetxt(original_pssm_outfile, original_pssm, fmt='%.4f') 
    
    reconstruct_samples_reshape = reconstruct_samples.reshape(reconstruct_samples.shape[0],reconstruct_samples.shape[1]*reconstruct_samples.shape[2])
    print "Generating gan feature for: ",pdb_name, " -----> Reconstruction error: ",reconstruction_test_error
    outfile = outputdir + '/'+pdb_name+'_gan.fea'
    np.savetxt(outfile, reconstruct_samples_reshape, fmt='%.4f') 
    
    content = "%s\t%.4f\n" % (pdb_name,reconstruction_test_error)
    with open(test_data_history_out, "a") as myfile:
        myfile.write(content)
                


##### running validation
sequence_file=open(val_datafile,'r').readlines() 

val_data_history_out = "%s/validation.reconstruction_history" % (outputdir)
chkdirs(val_data_history_out)
with open(val_data_history_out, "w") as myfile:
  myfile.write("PDB_name\tReconstructionError\n")

for i in xrange(len(sequence_file)):
    pdb_name = sequence_file[i].rstrip()
    val_featuredata_all=Vallist_data_keys[pdb_name]
    val_targets=Vallist_targets_keys[pdb_name]
    
    nb_test = len(val_targets)
    noise = np.random.uniform(-1, 1, (nb_test, latent_size))
    reconstruct_samples = generator.predict([noise, val_targets.reshape((-1, 1))], verbose=0)
    reconstruction_test_error = ((val_featuredata_all - reconstruct_samples) ** 2).mean()
    
    reconstruct_pssm = reconstruct_samples[:,(AA_win-1)/2,0:20] 
    original_pssm = val_featuredata_all[:,(AA_win-1)/2,0:20] 
    reconstruct_pssm_outfile = outputdir + '/'+pdb_name+'_gan.pssm_fea'
    np.savetxt(reconstruct_pssm_outfile, reconstruct_pssm, fmt='%.4f') 
    
    original_pssm_outfile = outputdir + '/'+pdb_name+'_orig.pssm_fea'
    np.savetxt(original_pssm_outfile, original_pssm, fmt='%.4f') 
    
    reconstruct_samples_reshape = reconstruct_samples.reshape(reconstruct_samples.shape[0],reconstruct_samples.shape[1]*reconstruct_samples.shape[2])
    print "Generating gan feature for: ",pdb_name, " -----> Reconstruction error: ",reconstruction_test_error
    outfile = outputdir + '/'+pdb_name+'_gan.fea'
    np.savetxt(outfile, reconstruct_samples_reshape, fmt='%.4f') 
    
    content = "%s\t%.4f\n" % (pdb_name,reconstruction_test_error)
    with open(val_data_history_out, "a") as myfile:
        myfile.write(content)



print "!!!! Results are saved in folder ", outputdir;