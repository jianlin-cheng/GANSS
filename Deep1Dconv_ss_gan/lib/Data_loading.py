# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:40:30 2017

@author: Jie Hou
"""
import os
import numpy as np

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)


def load_train_test_data_for_gan(data_list, feature_dir):
  import pickle
  
  #data_list ="/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/adj_dncon-train.lst"
  #feature_dir='/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win1/'
  sequence_file=open(data_list,'r').readlines() 
  data_all_dict = []
  seq_num=0;
  print "######### Loading training data\n\t"
  for i in xrange(len(sequence_file)):
      pdb_name = sequence_file[i].rstrip()
      print pdb_name, "..",
      featurefile = feature_dir + '/' + pdb_name + '.fea'
      if not os.path.isfile(featurefile):
                  print "feature file not exists: ",featurefile, " pass!"
                  continue           
      
      featuredata = np.loadtxt(featurefile) #(169, 23) # 3+ 20
      data_all_dict.append(featuredata)
      seq_num +=1
  data_all_dict =  np.concatenate(data_all_dict)
  print "number of sequences: ",seq_num;
  print "Shape of data_all_dict: ",data_all_dict.shape;
  return data_all_dict
