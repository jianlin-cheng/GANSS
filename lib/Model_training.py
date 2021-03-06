# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os
from Model_construct import build_generator,build_discriminator,DeepCov_SS_with_paras,build_discriminator_postGAN,build_discriminator_variant1D,build_generator_variant1D,build_discriminator_variant1D,build_generator_variant1D_V2,build_discriminator_NN,build_generator_NN,build_discriminator_postGAN_variant1D
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

from Custom_class import K_max_pooling1d


def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)


def DeepSS_1dconv_gan_train_win_filter_layer_opt(data_all_dict,testdata_all_dict,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,batch_size,win_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5):
    import numpy as np
    
    feature_num=0; # the number of features for each residue
    
    train_labels = data_all_dict[:,0:3] ## (1,0,0)
    train_labels_convert = np.argmax(train_labels, axis=1) ## need convert to number for gan
    train_feature = data_all_dict[:,3:]
    train_samples=train_feature.shape[0]
    if train_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of train features (%i) not be divided by residue num (%i) " % (train_feature.shape[1],AA_win))
    train_feature_reshape = train_feature.reshape((train_samples,AA_win,train_feature.shape[1]/AA_win))
    
    print "################ Train Feature number: " , train_feature.shape[1]/AA_win
    print "################ Train Residue number: ",AA_win;
    print "################ Shape of training data: ",train_feature_reshape.shape;
    
    feature_num = train_feature.shape[1]/AA_win
    
    test_labels = testdata_all_dict[:,0:3]
    test_labels_convert = np.argmax(test_labels, axis=1) ## need convert to number for gan
    test_feature = testdata_all_dict[:,3:]
    test_samples=test_feature.shape[0]
    if test_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of test features (%i) not be divided by residue num (%i) " % (test_feature.shape[1],AA_win))
    test_feature_reshape = test_feature.reshape((test_samples,AA_win,test_feature.shape[1]/AA_win))
    
    print "\n################ Test Feature number: ",test_feature.shape[1]/AA_win;
    print "################ Test Residue number: ",AA_win;
    print "################ Shape of testing data: ",test_feature_reshape.shape;    
    
    
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
 
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    nb_layers_generator = nb_layers_generator 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = nb_filters # this is for discriminator
    nb_filters_generator = fea_num # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    

    
    ### Define the model 
    model_generator_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_out= "%s/model-train-generator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_weight_out= "%s/model-train-generator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    if os.path.exists(model_discriminator_out):
        print "######## Loading existing discriminator model ",model_discriminator_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_out)
    else:    
        # build the discriminator
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers_discriminator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator(AA_win,nb_filters,nb_layers_discriminator,win_array,fea_num,n_class)
      
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    if os.path.exists(model_discriminator_weight_out):
    	print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
    	discriminator.load_weights(model_discriminator_weight_out)
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    else:
    	print "######## Setting initial discriminator weights";
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    
    # build the generator
    if os.path.exists(model_generator_out):
        print "######## Loading existing generator model ",model_generator_out;
        generator=Sequential()
        generator=load_model(model_generator_out)
    else:
        print "\n\n#### Start initializing generator: ";
        print "         latent_size: ",latent_size;
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters_generator;
        print "         nb_layers: ",nb_layers_generator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        generator = build_generator(latent_size,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class)
    
    if os.path.exists(model_generator_weight_out):
    	print "######## Loading existing generator weights ",model_generator_weight_out;
    	generator.load_weights(model_generator_weight_out)
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
    else:
    	print "######## Setting initial generator weights";
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
        
    print "\n\n#### Summary of Generator: ";
    print(generator.summary())
    
    latent = Input(shape=(latent_size, ))
    ss_class = Input(shape=(1,), dtype='int32')
    
    # get a fake image
    fake = generator([latent, ss_class])
    
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, ss_class], output=[fake, aux])
    #combined.summary()
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    
    print "\n\n#### Summary of combined model: ";
    print(combined.summary())
    
    
    X_train = train_feature_reshape
    y_train = train_labels_convert
    
    X_test = test_feature_reshape
    y_test = test_labels_convert
    
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]
    
    
    print "\n\n#### Start training GAN: ";
    #print "         X_train: ",X_train.shape;
    #print "         y_train: ",y_train.shape;
    #print "         X_test: ",X_test.shape;
    #print "         y_test: ",y_test.shape;
    
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
                     
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        if X_train.shape[0] % batch_size != 0:
          nb_batches = int(X_train.shape[0] / batch_size) + 1
        else:
          nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        
        for index in range(nb_batches):
            ##progress_bar.update(index)
            
            # get a batch of real images
            sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            
            batch_size_indata = batch_size
            if sample_batch.shape[0] < batch_size:
                batch_size_indata = sample_batch.shape[0]  ## means sample less than batch size
            
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size_indata, latent_size))   
            # sample some labels from p_c
            sampled_labels = np.random.randint(0, n_class, batch_size_indata)
            
            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_samples = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            
            X = np.concatenate((sample_batch, generated_samples))
            y = np.array([1] * batch_size_indata + [0] * batch_size_indata)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
            
            
            #### generate reconstruction error
            reconstruct_samples = generator.predict(
                [noise, label_batch.reshape((-1, 1))], verbose=0)
            reconstruction_error = ((sample_batch - reconstruct_samples) ** 2).mean()
            reconstruct_gen_loss.append(reconstruction_error)
            
            
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
            
            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size_indata, latent_size))
            sampled_labels = np.random.randint(0, n_class, 2 * batch_size_indata)
            
            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size_indata)
            
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))
        
        print('\nTesting for epoch {}:'.format(epoch + 1))
        # evaluate the testing loss here
        
        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))
        
        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, n_class, nb_test)
        generated_samples = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)
        
        X = np.concatenate((X_test, generated_samples))
        
        
        #### generate reconstruction error using true label
        reconstruct_samples = generator.predict(
            [noise, y_test.reshape((-1, 1))], verbose=0)
        reconstruction_test_error = ((X_test - reconstruct_samples) ** 2).mean()
        
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
        
        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)
        
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, n_class, 2 * nb_test)
        
        trick = np.ones(2 * nb_test)
        
        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        
        reconstruction_train_loss = np.mean(np.array(reconstruct_gen_loss), axis=0)
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['reconstrution'].append(reconstruction_train_loss)
        
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        test_history['reconstrution'].append(reconstruction_test_error)
        
        
        # save weights every epoch
        #generator_weigth_out= "%s/params_generator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)
        
        generator.save(model_generator_out)  
        generator.save_weights(model_generator_weight_out, True)
                
        generator_model_out_tmp= "%s/%s/params_generator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(generator_model_out_tmp)      
        generator.save(generator_model_out_tmp)
        
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
                      
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        test_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
                    myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        train_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
                    myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        val_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            generator.save_weights(model_generator_best_weight_out, True)        
            generator.save(model_generator_best_out)
            
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        
        if epoch % 10== 0 and epoch > 0:
            args_str ="perl "+ lib_dir +"/visualize_training_score_gan.pl "  + CV_dir
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            
            summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
            check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
            found = 0
            while (found == 0):
                print "Checking file ",check_file
                time.sleep(15) 
                if os.path.exists(check_file):
                  found = 1
            print "Temporary visualization saved to file ",summary_file
            
            image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
            
            args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            found = 0
            while (found == 0):
                print "Checking file ",image_file
                time.sleep(15) 
                if os.path.exists(image_file):
                  found = 1
            print "Temporary visualization saved to file ",image_file
                            
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (val)',
                             *test_history['generator'][-1]))
                             
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))                             
        print(ROW_FMT.format('discriminator (val)',
                             *test_history['discriminator'][-1]))
                             
        print 'Reconstruction Error (train): ', train_history['reconstrution'][-1]
        print 'Reconstruction Error (val): ',test_history['reconstrution'][-1]
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('\n{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                  'component', *discriminator.metrics_names))
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}\n'
          myfile.write(ROW_FMT.format('generator (train)',
                               *train_history['generator'][-1]))
          myfile.write(ROW_FMT.format('generator (test)',
                               *test_history['generator'][-1]))
                               
          myfile.write(ROW_FMT.format('discriminator (train)',
                               *train_history['discriminator'][-1]))                             
          myfile.write(ROW_FMT.format('discriminator (val)',
                               *test_history['discriminator'][-1]))       
          myfile.write("Reconstruction Error (train): %.5f\n" %  train_history['reconstrution'][-1])
          myfile.write("Reconstruction Error (val): %.5f\n" % test_history['reconstrution'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    generator.load_weights(model_generator_best_weight_out)
    discriminator.load_weights(model_discriminator_best_weight_out)
    generator.save(model_generator_out)  
    generator.save_weights(model_generator_weight_out, True)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    pickle.dump({'train': train_history, 'test': test_history},
        open('acgan-history.pkl', 'wb'))



def DeepSS_1dNN_gan_train_win_filter_layer_opt(data_all_dict,testdata_all_dict,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,batch_size,nodes_array,win_array,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5):
    import numpy as np
    
    feature_num=0; # the number of features for each residue
    
    train_labels = data_all_dict[:,0:3] ## (1,0,0)
    train_labels_convert = np.argmax(train_labels, axis=1) ## need convert to number for gan
    train_feature = data_all_dict[:,3:]
    train_samples=train_feature.shape[0]
    if train_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of train features (%i) not be divided by residue num (%i) " % (train_feature.shape[1],AA_win))
    train_feature_reshape = train_feature.reshape((train_samples,train_feature.shape[1]))
    
    print "################ Train Feature number: " , train_feature.shape[1]/AA_win
    print "################ Train Residue number: ",AA_win;
    print "################ Shape of training data: ",train_feature_reshape.shape;
    
    feature_num = train_feature.shape[1]/AA_win
    
    test_labels = testdata_all_dict[:,0:3]
    test_labels_convert = np.argmax(test_labels, axis=1) ## need convert to number for gan
    test_feature = testdata_all_dict[:,3:]
    test_samples=test_feature.shape[0]
    if test_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of test features (%i) not be divided by residue num (%i) " % (test_feature.shape[1],AA_win))
    test_feature_reshape = test_feature.reshape((test_samples,test_feature.shape[1]))
    
    print "\n################ Test Feature number: ",test_feature.shape[1]/AA_win;
    print "################ Test Residue number: ",AA_win;
    print "################ Shape of testing data: ",test_feature_reshape.shape;    
    
    
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
            Trainlist_data_keys[pdb_name]=train_feature_tmp.reshape(train_feature_tmp.shape[0],train_feature_tmp.shape[1])
        
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
            Testlist_data_keys[pdb_name]=test_feature_tmp.reshape(test_feature_tmp.shape[0],test_feature_tmp.shape[1])
        
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
            Vallist_data_keys[pdb_name]=val_feature_tmp.reshape(val_feature_tmp.shape[0],val_feature_tmp.shape[1])
        
        if pdb_name in Vallist_targets_keys:
            print "Duplicate pdb name %s in Val list " % pdb_name
        else:
            Vallist_targets_keys[pdb_name]=val_labels_tmp
 
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    nb_layers_generator = nb_layers_generator 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    #nb_filters = nb_filters # no need for NN discriminator
    nb_filters_generator = fea_num # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    
    ### Define the model 
    model_generator_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_out= "%s/model-train-generator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_weight_out= "%s/model-train-generator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    
    if os.path.exists(model_discriminator_out):
    	print "######## Loading existing discriminator model ",model_discriminator_out;
    	discriminator=Sequential()
    	discriminator=load_model(model_discriminator_out, custom_objects={'K_max_pooling1d': K_max_pooling1d})   
    else:    
    	# build the discriminator 
    	print "\n\n#### Start initializing discriminator: ";
    	print "         AA_win: ",AA_win;
    	print "         nb_layers: ",nb_layers_discriminator;
    	print "         nodes_array: ",nodes_array;
    	print "         fea_num: ",fea_num;
    	print "         n_class: ",n_class;
    	discriminator = build_discriminator_NN(AA_win,nb_layers_discriminator,nodes_array,fea_num,n_class)
    
    
    if os.path.exists(model_discriminator_weight_out):
    	print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
    	discriminator.load_weights(model_discriminator_weight_out)
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    else:
    	print "######## Setting initial discriminator weights";
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    
    
    
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    
    if os.path.exists(model_generator_out):
    	print "######## Loading existing generator model ",model_generator_out;
    	generator=Sequential()
    	generator=load_model(model_generator_out, custom_objects={'K_max_pooling1d': K_max_pooling1d})   
    else:    
    	# build the generator 
    	print "\n\n#### Start initializing generator: ";
    	print "         latent_size: ",latent_size;
    	print "         AA_win: ",AA_win;
    	print "         nb_filters: ",nb_filters_generator;
    	print "         nb_layers: ",nb_layers_generator;
    	print "         win_array: ",win_array;
    	print "         fea_num: ",fea_num;
    	print "         n_class: ",n_class;
    	generator = build_generator_NN(latent_size,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class)
    
    if os.path.exists(model_generator_weight_out):
    	print "######## Loading existing generator weights ",model_generator_weight_out;
    	generator.load_weights(model_generator_weight_out)
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
    else:
    	print "######## Setting initial generator weights";
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
    
    
    print "\n\n#### Summary of generator: ";
    print(generator.summary())
    
    latent = Input(shape=(latent_size, ))
    ss_class = Input(shape=(1,), dtype='int32')
    
    # get a fake image
    fake = generator([latent, ss_class])
    
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, ss_class], output=[fake, aux])
    #combined.summary()
    combined.compile(
    	optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
    	loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    
    print "\n\n#### Summary of combined model: ";
    print(combined.summary())

    
    X_train = train_feature_reshape
    y_train = train_labels_convert
    
    X_test = test_feature_reshape
    y_test = test_labels_convert
    
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]
    
    
    print "\n\n#### Start training GAN: ";
    #print "         X_train: ",X_train.shape;
    #print "         y_train: ",y_train.shape;
    #print "         X_test: ",X_test.shape;
    #print "         y_test: ",y_test.shape;
    
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
                     
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        if X_train.shape[0] % batch_size != 0:
          nb_batches = int(X_train.shape[0] / batch_size) + 1
        else:
          nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        
        for index in range(nb_batches):
            #progress_bar.update(index)
            
            # get a batch of real images
            sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            
            batch_size_indata = batch_size
            if sample_batch.shape[0] < batch_size:
                batch_size_indata = sample_batch.shape[0]  ## means sample less than batch size
            
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size_indata, latent_size))   
            # sample some labels from p_c
            sampled_labels = np.random.randint(0, n_class, batch_size_indata)
            
            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_samples = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            
            X = np.concatenate((sample_batch, generated_samples))
            y = np.array([1] * batch_size_indata + [0] * batch_size_indata)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
            
            
            #### generate reconstruction error
            reconstruct_samples = generator.predict(
                [noise, label_batch.reshape((-1, 1))], verbose=0)
            reconstruction_error = ((sample_batch - reconstruct_samples) ** 2).mean()
            reconstruct_gen_loss.append(reconstruction_error)
            
            
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
            
            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size_indata, latent_size))
            sampled_labels = np.random.randint(0, n_class, 2 * batch_size_indata)
            
            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size_indata)
            
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))
        
        print('\nTesting for epoch {}:'.format(epoch + 1))
        # evaluate the testing loss here
        
        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))
        
        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, n_class, nb_test)
        generated_samples = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)
        
        X = np.concatenate((X_test, generated_samples))
        
        
        #### generate reconstruction error using true label
        reconstruct_samples = generator.predict(
            [noise, y_test.reshape((-1, 1))], verbose=0)
        reconstruction_test_error = ((X_test - reconstruct_samples) ** 2).mean()
        
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
        
        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)
        
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, n_class, 2 * nb_test)
        
        trick = np.ones(2 * nb_test)
        
        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        
        reconstruction_train_loss = np.mean(np.array(reconstruct_gen_loss), axis=0)
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['reconstrution'].append(reconstruction_train_loss)
        
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        test_history['reconstrution'].append(reconstruction_test_error)
        
        
         # save weights every epoch
        #generator_weigth_out= "%s/params_generator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)
        
        generator.save(model_generator_out)  
        generator.save_weights(model_generator_weight_out, True)
                
        generator_model_out_tmp= "%s/%s/params_generator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(generator_model_out_tmp)      
        generator.save(generator_model_out_tmp)
        
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
                      
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        test_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
                    myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        train_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
                    myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result[1]
            predict_val_truth=result[0]
              
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
        
        val_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            generator.save_weights(model_generator_best_weight_out, True)        
            generator.save(model_generator_best_out)
            
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        
        if epoch % 10== 0 and epoch > 0:
            args_str ="perl "+ lib_dir +"/visualize_training_score_gan.pl "  + CV_dir
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            
            summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
            check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
            found = 0
            while (found == 0):
                print "Checking file ",check_file
                time.sleep(15) 
                if os.path.exists(check_file):
                  found = 1
            print "Temporary visualization saved to file ",summary_file
            
            image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
            
            args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            found = 0
            while (found == 0):
                print "Checking file ",image_file
                time.sleep(15) 
                if os.path.exists(image_file):
                  found = 1
            print "Temporary visualization saved to file ",image_file
                            
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (val)',
                             *test_history['generator'][-1]))
                             
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))                             
        print(ROW_FMT.format('discriminator (val)',
                             *test_history['discriminator'][-1]))
                             
        print 'Reconstruction Error (train): ', train_history['reconstrution'][-1]
        print 'Reconstruction Error (val): ',test_history['reconstrution'][-1]
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('\n{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                  'component', *discriminator.metrics_names))
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}\n'
          myfile.write(ROW_FMT.format('generator (train)',
                               *train_history['generator'][-1]))
          myfile.write(ROW_FMT.format('generator (test)',
                               *test_history['generator'][-1]))
                               
          myfile.write(ROW_FMT.format('discriminator (train)',
                               *train_history['discriminator'][-1]))                             
          myfile.write(ROW_FMT.format('discriminator (val)',
                               *test_history['discriminator'][-1]))       
          myfile.write("Reconstruction Error (train): %.5f\n" %  train_history['reconstrution'][-1])
          myfile.write("Reconstruction Error (val): %.5f\n" % test_history['reconstrution'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    generator.load_weights(model_generator_best_weight_out)
    discriminator.load_weights(model_discriminator_best_weight_out)
    generator.save(model_generator_out)  
    generator.save_weights(model_generator_weight_out, True)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    
    pickle.dump({'train': train_history, 'test': test_history},
        open('acgan-history.pkl', 'wb'))




#### this is traditional window-based CNN method
def DeepSS_1dconv_fixed_window_train_win_filter_layer_opt(data_all_dict,testdata_all_dict,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,batch_size,win_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5,postGAN='None'):
    import numpy as np
    
    feature_num=0; # the number of features for each residue
    
    train_labels = data_all_dict[:,0:3] ## (1,0,0)
    train_labels_convert = np.argmax(train_labels, axis=1) ## need convert to number for gan
    train_feature = data_all_dict[:,3:]
    train_samples=train_feature.shape[0]
    if train_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of train features (%i) not be divided by residue num (%i) " % (train_feature.shape[1],AA_win))
    train_feature_reshape = train_feature.reshape((train_samples,AA_win,train_feature.shape[1]/AA_win))
    
    print "################ Train Feature number: " , train_feature.shape[1]/AA_win
    print "################ Train Residue number: ",AA_win;
    print "################ Shape of training data: ",train_feature_reshape.shape;
    
    feature_num = train_feature.shape[1]/AA_win
    
    test_labels = testdata_all_dict[:,0:3]
    test_labels_convert = np.argmax(test_labels, axis=1) ## need convert to number for gan
    test_feature = testdata_all_dict[:,3:]
    test_samples=test_feature.shape[0]
    if test_feature.shape[1] % AA_win != 0:
        raise Exception("The amount of test features (%i) not be divided by residue num (%i) " % (test_feature.shape[1],AA_win))
    test_feature_reshape = test_feature.reshape((test_samples,AA_win,test_feature.shape[1]/AA_win))
    
    print "\n################ Test Feature number: ",test_feature.shape[1]/AA_win;
    print "################ Test Residue number: ",AA_win;
    print "################ Shape of testing data: ",test_feature_reshape.shape;    
    
    
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
 
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = nb_filters # this is for discriminator
    nb_filters_generator = fea_num # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    ### Define the model 
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    model_discriminator_postGAN_initial= "%s/model-train-discriminator-postGAN-init-%s.hdf5" % (CV_dir,model_prefix)
    
    if postGAN != 'None':
        if os.path.exists(postGAN):
          print "######## Loading existing postGAN discriminator model ",postGAN;
          discriminator=Sequential()
          discriminator=load_model(postGAN)
        else:
          raise Exception("Failed to find postGAN model (%s) " % postGAN)
    elif os.path.exists(model_discriminator_out):
        print "######## Loading existing discriminator model ",model_discriminator_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_out)
    else:    
        # build the discriminator
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers_discriminator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator_postGAN(AA_win,nb_filters,nb_layers_discriminator,win_array,fea_num,n_class)

    if os.path.exists(model_discriminator_weight_out):
    	print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
    	discriminator.load_weights(model_discriminator_weight_out)
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['sparse_categorical_crossentropy'])
    else:
    	print "######## Setting initial discriminator weights";
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['sparse_categorical_crossentropy'])
    
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    X_train = train_feature_reshape
    y_train = train_labels_convert

    X_test = test_feature_reshape
    y_test = test_labels_convert

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]
    
    
    print "\n\n#### Start training discriminator: ";
    #print "         X_train: ",X_train.shape;
    #print "         y_train: ",y_train.shape;
    #print "         X_test: ",X_test.shape;
    #print "         y_test: ",y_test.shape;
    
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("AA_window\tEpoch_outside\tAccuracy\n")
    
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
    
                     
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        if X_train.shape[0] % batch_size != 0:
          nb_batches = int(X_train.shape[0] / batch_size) + 1
        else:
          nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        
        for index in range(nb_batches):
            #progress_bar.update(index)
            # generate a new batch of noise
            
            # get a batch of real images
            sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(sample_batch, label_batch))
        
        print('\nTesting for epoch {}:'.format(epoch + 1))
        # evaluate the testing loss here
        
        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X_test, y_test, verbose=False)
        
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        # generate an epoch report on performance
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
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
            predict_val=result
              
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
        
        test_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
                    myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result
              
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
        
        train_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
                    myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            predict_val=result
              
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
        
        val_acc_history_content = "%i\t%i\t%.4f\n" % (AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        """
        if epoch % 10== 0 and epoch > 0:
            args_str ="perl "+ lib_dir +"/visualize_training_score_gan.pl "  + CV_dir
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            
            summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
            check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
            found = 0
            while (found == 0):
                print "Checking file ",check_file
                time.sleep(15) 
                if os.path.exists(check_file):
                  found = 1
            print "Temporary visualization saved to file ",summary_file
            
            image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
            
            args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            found = 0
            while (found == 0):
                print "Checking file ",image_file
                time.sleep(15) 
                if os.path.exists(image_file):
                  found = 1
            print "Temporary visualization saved to file ",image_file
      
        """           
        print('Training summary:')
        print('-' * 65)
                             
        print "discriminator loss (train): %.3f" % train_history['discriminator'][-1] 
        print "discriminator loss (val): %.3f" % test_history['discriminator'][-1]
        
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('Training summary:')
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          myfile.write("discriminator (train): %.3f" % train_history['discriminator'][-1])                             
          myfile.write("discriminator (val): %.3f" % test_history['discriminator'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    discriminator.load_weights(model_discriminator_best_weight_out)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    
    pickle.dump({'train': train_history, 'test': test_history},
        open('acgan-history.pkl', 'wb'))


### this is not working well, need generate the fake residue by residue, check updated function  DeepSS_1dconv_gan_variant_train_win_filter_layer_opt_latest
def DeepSS_1dconv_gan_variant_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,interval_len,seq_end,batch_size,win_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5):
    import numpy as np
    start=30 ## since in the build_discriminator_variant1D, I used kmax30, so the sequence < 30 should be removed
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    Test_data_keys = dict()
    Test_targets_keys = dict()
    feature_num=0; # the number of features for each residue
    
    for key in data_all_dict_padding.keys():
        if key > end: # run first model on 100 at most
          continue
        #print "\n### Loading sequence length :", key
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_labels = trainfeaturedata[:,:,0:3]
        train_labels_convert = np.argmax(train_labels, axis=2) ## need convert to number for gan
        train_feature = trainfeaturedata[:,:,3:]
        feature_num=train_feature.shape[2]/AA_win
        if seq_len in testdata_all_dict_padding:
          testfeaturedata = testdata_all_dict_padding[seq_len]
          #print "Loading test dataset "
        else:
          testfeaturedata = trainfeaturedata
          print "\n\n##Warning: Setting training dataset as testing dataset \n\n"
        
        
        test_labels = testfeaturedata[:,:,0:3]
        test_labels_convert = np.argmax(test_labels, axis=2) ## need convert to number for gan
        test_feature = testfeaturedata[:,:,3:]    
        sequence_length = seq_len
        
        if seq_len in Train_data_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_data_keys[seq_len]=(train_feature)
          
        if seq_len in Train_targets_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_targets_keys[seq_len]=train_labels_convert        
        #processing test data 
        if seq_len in Test_data_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_data_keys[seq_len]=test_feature 
        
        if seq_len in Test_targets_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_targets_keys[seq_len]=test_labels_convert
    
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
        train_labels = featuredata[:,0:3]#(169, 3)
        train_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Trainlist_data_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_data_keys[pdb_name]=train_feature.reshape(1,train_feature.shape[0],train_feature.shape[1])
        
        if pdb_name in Trainlist_targets_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_targets_keys[pdb_name]=train_labels.reshape(1,train_labels.shape[0],train_labels.shape[1])
    
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
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Testlist_data_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_data_keys[pdb_name]=test_feature.reshape(1,test_feature.shape[0],test_feature.shape[1])
        
        if pdb_name in Testlist_targets_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_targets_keys[pdb_name]=test_labels.reshape(1,test_labels.shape[0],test_labels.shape[1])
    
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
        val_labels = featuredata[:,0:3]#(169, 3)
        val_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Vallist_data_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_data_keys[pdb_name]=val_feature.reshape(1,val_feature.shape[0],val_feature.shape[1])
        
        if pdb_name in Vallist_targets_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_targets_keys[pdb_name]=val_labels.reshape(1,val_labels.shape[0],val_labels.shape[1])
    
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    nb_layers_generator = nb_layers_generator 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = nb_filters # this is for discriminator
    nb_filters_generator = fea_num * AA_win # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    ### Define the model 
    model_generator_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_out= "%s/model-train-generator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_weight_out= "%s/model-train-generator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    if os.path.exists(model_discriminator_out):
        print "######## Loading existing discriminator model ",model_discriminator_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_best_out, custom_objects={'K_max_pooling1d': K_max_pooling1d})   
      
    else:    
        # build the discriminator 
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers_discriminator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator_variant1D(nb_filters,nb_layers_discriminator,win_array,AA_win,fea_num)
    
    if os.path.exists(model_discriminator_weight_out):
    	print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
    	discriminator.load_weights(model_discriminator_weight_out)
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    else:
    	print "######## Setting initial discriminator weights";
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    # build the generator
    if os.path.exists(model_generator_best_out):
        print "######## Loading existing generator model ",model_generator_best_out;
        generator=Sequential()
        generator=load_model(model_generator_best_out)
    else:
        print "\n\n#### Start initializing generator: ";
        print "         latent_size: ",latent_size;
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters_generator;
        print "         nb_layers: ",nb_layers_generator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        generator = build_generator_variant1D(latent_size,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class)
    
    
    if os.path.exists(model_generator_weight_out):
    	print "######## Loading existing generator weights ",model_generator_weight_out;
    	generator.load_weights(model_generator_weight_out)
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
    else:
    	print "######## Setting initial generator weights";
    	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
    
    
    print "\n\n#### Summary of Generator: ";
    print(generator.summary())
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
    
    
    ### train by length   
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        epoch_test_loss = []
        
        for key in data_all_dict_padding.keys():
            if key <start: # run first model on 100 at most
              continue
            if key > end: # run first model on 100 at most
              continue
            #print '### Loading sequence length :', key
            seq_len=key
            
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            test_featuredata_all=Test_data_keys[seq_len]
            test_targets=Test_targets_keys[seq_len]
            #print "Train shape: ",train_featuredata_all.shape, " in outside epoch ", epoch 
            #print "Test shape: ",test_featuredata_all.shape, " in outside epoch ", epoch
                
            latent = Input(shape=(seq_len,latent_size))
            ss_class = Input(shape=(seq_len,), dtype='int32')
            
            # get a fake image
            fake = generator([latent, ss_class])
            
            # we only want to be able to train generation for the combined model
            discriminator.trainable = False
            fake, aux = discriminator(fake)
            combined = Model(input=[latent, ss_class], output=[fake, aux])
            #combined.summary()
            combined.compile(
              optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
              loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
            )
            
            #print "\n\n#### Summary of combined model: ";
            #print(combined.summary())
            
            
            X_train = train_featuredata_all
            y_train = train_targets
            
            X_test = test_featuredata_all
            y_test = test_targets
            
            nb_train, nb_test = X_train.shape[0], X_test.shape[0]
            
            if X_train.shape[0] % batch_size != 0:
              nb_batches = int(X_train.shape[0] / batch_size) + 1
            else:
              nb_batches = int(X_train.shape[0] / batch_size)
            
            #print "\n\n#### Start training GAN: ";
            #print "         X_train: ",X_train.shape;
            #print "         y_train: ",y_train.shape;
            #print "         X_test: ",X_test.shape;
            #print "         y_test: ",y_test.shape;
            #print "         Len: ",seq_len;
            #print "         batch_size: ",batch_size;
            #print "         nb_batches: ",nb_batches;
            progress_bar = Progbar(target=nb_batches)
            for index in range(nb_batches):
                #progress_bar.update(index)
                
                # get a batch of real images
                sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]
                #print "sample_batch.shape: ",sample_batch.shape
                #print "label_batch.shape: ",label_batch.shape
                batch_size_indata = batch_size
                if sample_batch.shape[0] < batch_size:
                    batch_size_indata = sample_batch.shape[0]  ## means sample less than batch size
                
                # generate a new batch of noise
                noise = np.random.uniform(-1, 1, (batch_size_indata, seq_len, latent_size))
                
                # sample some labels from p_c
                sampled_labels = np.random.randint(0, n_class, (batch_size_indata,seq_len))
                
                # generate a batch of fake images, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (batch_size, 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_samples = generator.predict(
                  [noise, sampled_labels.reshape((batch_size_indata, seq_len))], verbose=0)
                
                X = np.concatenate((sample_batch, generated_samples))
                y = np.array([1] * batch_size_indata + [0] * batch_size_indata)
                
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0) #(20,255)
                aux_y = aux_y.reshape((aux_y.shape[0],aux_y.shape[1],1))#(20,255,1)
                #aux_y_categorical = (np.arange(aux_y.max()+1) == aux_y[...,None]).astype(int)  # since use sparse_categorical_crossentropy, no need for one-hot encoding
                
                #### generate reconstruction error
                reconstruct_samples = generator.predict(
                  [noise, label_batch.reshape((batch_size_indata, seq_len))], verbose=0)
                reconstruction_error = ((sample_batch - reconstruct_samples) ** 2).mean()
                reconstruct_gen_loss.append(reconstruction_error)
                #print "in: reconstruct_gen_loss: ",reconstruct_gen_loss
                
                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
                #print "in: epoch_disc_loss: ",epoch_disc_loss
                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator
                noise = np.random.uniform(-1, 1, (2 * batch_size_indata, seq_len, latent_size))
                sampled_labels = np.random.randint(0, n_class, (2 * batch_size_indata,seq_len))#(20,255)
                sampled_labels = sampled_labels.reshape((sampled_labels.shape[0],sampled_labels.shape[1],1))#(20,255,1)
                
                #sampled_labels_categorical = (np.arange(sampled_labels.max()+1) == sampled_labels[...,None]).astype(int)
                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = np.ones(2 * batch_size_indata)
                
                epoch_gen_loss.append(combined.train_on_batch(
                  [noise, sampled_labels.reshape((2 * batch_size_indata, seq_len))], [trick, sampled_labels]))
                #print "in: epoch_gen_loss: ",epoch_gen_loss
            
            #print('\nTesting for epoch {}:'.format(epoch + 1))
            # evaluate the testing loss here
            
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (nb_test, seq_len,latent_size))
            
            # sample some labels from p_c and generate images from them
            sampled_labels = np.random.randint(0, n_class, (nb_test,seq_len))
            generated_samples = generator.predict(
              [noise, sampled_labels.reshape((nb_test, seq_len))], verbose=False)
            
            X = np.concatenate((X_test, generated_samples))
            
            
            #### generate reconstruction error using true label
            reconstruct_samples = generator.predict(
              [noise, y_test.reshape((nb_test, seq_len))], verbose=0)
            reconstruction_test_error = ((X_test - reconstruct_samples) ** 2).mean()
            
            y = np.array([1] * nb_test + [0] * nb_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0) #(20,255)
            aux_y = aux_y.reshape((aux_y.shape[0],aux_y.shape[1],1))#(20,255,1)
            
            #aux_y_categorical = (np.arange(aux_y.max()+1) == aux_y[...,None]).astype(int)
            # see if the discriminator can figure itself out...
            discriminator_test_loss = discriminator.evaluate(
              X, [y, aux_y], verbose=False)
            
            # make new noise
            noise = np.random.uniform(-1, 1, (2 * nb_test, seq_len, latent_size))
            sampled_labels = np.random.randint(0, n_class, (2 * nb_test,seq_len)) #(20,255)
            sampled_labels = sampled_labels.reshape((sampled_labels.shape[0],sampled_labels.shape[1],1))#(20,255,1)
            #sampled_labels_categorical = (np.arange(sampled_labels.max()+1) == sampled_labels[...,None]).astype(int)
            
            trick = np.ones(2 * nb_test)
            
            epoch_test_loss.append(combined.evaluate(
              [noise, sampled_labels.reshape((2 * nb_test, seq_len))],
              [trick, sampled_labels], verbose=False))
        
        #print "Average means"
        #print "epoch_gen_loss: ",epoch_gen_loss
        #print "epoch_disc_loss: ",epoch_disc_loss
        #print "epoch_test_loss: ",epoch_test_loss
        #print "reconstruct_gen_loss: ",reconstruct_gen_loss
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_test_loss = np.mean(np.array(epoch_test_loss), axis=0)
        reconstruction_train_loss = np.mean(np.array(reconstruct_gen_loss), axis=0)
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['reconstrution'].append(reconstruction_train_loss)
        
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        test_history['reconstrution'].append(reconstruction_test_error)
        
        
         # save weights every epoch
        #generator_weigth_out= "%s/params_generator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)
        
        generator.save(model_generator_out)  
        generator.save_weights(model_generator_weight_out, True)
                
        generator_model_out_tmp= "%s/%s/params_generator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(generator_model_out_tmp)      
        generator.save(generator_model_out_tmp)
        
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
        
            
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
        print "start evaluating test"
        for i in xrange(len(sequence_file)):
            pdb_name = sequence_file[i].rstrip()
            if pdb_name not in Testlist_data_keys:
                    print 'removing ',pdb_name
                    continue
            test_featuredata_all=Testlist_data_keys[pdb_name]
            test_targets=Testlist_targets_keys[pdb_name]
            result= discriminator.predict([test_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            #print "pdb_name: ",pdb_name;
            #print "test_featuredata_all: ",test_featuredata_all.shape
            #print "predict_val: ",predict_val.shape;
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            test_labels_convert = np.argmax(test_targets, axis=2).ravel()
            
            test_acc +=float(sum(test_labels_convert == preds_convert))/len(test_labels_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del test_featuredata_all
            del test_targets
        
        test_acc /= acc_num 
        
        test_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
              myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Trainlist_data_keys:
                    continue
            train_featuredata_all=Trainlist_data_keys[pdb_name]
            train_targets=Trainlist_targets_keys[pdb_name]
            result= discriminator.predict([train_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            train_targets_convert = np.argmax(train_targets, axis=2).ravel()
            train_acc +=float(sum(train_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del train_featuredata_all
            del train_targets
        
        train_acc /= acc_num
        
        train_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
              myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Vallist_data_keys:
                    continue
            val_featuredata_all=Vallist_data_keys[pdb_name]
            val_targets=Vallist_targets_keys[pdb_name]
            result= discriminator.predict([val_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            val_targets_convert = np.argmax(val_targets, axis=2).ravel()
            
            val_acc +=float(sum(val_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del val_featuredata_all
            del val_targets
        
        val_acc /= acc_num
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
              myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            generator.save_weights(model_generator_best_weight_out, True)        
            generator.save(model_generator_best_out)
            
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        
        if epoch % 10== 0 and epoch > 0:
          args_str ="perl "+ lib_dir +"/visualize_training_score_variablegan.pl "  + CV_dir
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          
          summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
          check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
          found = 0
          while (found == 0):
            print "Checking file ",check_file
            time.sleep(15) 
            if os.path.exists(check_file):
              found = 1
          print "Temporary visualization saved to file ",summary_file
          
          image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
          
          args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          found = 0
          while (found == 0):
              print "Checking file ",image_file
              time.sleep(15) 
              if os.path.exists(image_file):
                found = 1
          print "Temporary visualization saved to file ",image_file
        
              
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
          'component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                   *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (val)',
                   *test_history['generator'][-1]))
                   
        print(ROW_FMT.format('discriminator (train)',
                   *train_history['discriminator'][-1]))                             
        print(ROW_FMT.format('discriminator (val)',
                   *test_history['discriminator'][-1]))
                   
        print 'Reconstruction Error (train): ', train_history['reconstrution'][-1]
        print 'Reconstruction Error (val): ',test_history['reconstrution'][-1]
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('\n{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
              'component', *discriminator.metrics_names))
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}\n'
          myfile.write(ROW_FMT.format('generator (train)',
                     *train_history['generator'][-1]))
          myfile.write(ROW_FMT.format('generator (test)',
                     *test_history['generator'][-1]))
                     
          myfile.write(ROW_FMT.format('discriminator (train)',
                     *train_history['discriminator'][-1]))                             
          myfile.write(ROW_FMT.format('discriminator (val)',
                     *test_history['discriminator'][-1]))       
          myfile.write("Reconstruction Error (train): %.5f\n" %  train_history['reconstrution'][-1])
          myfile.write("Reconstruction Error (val): %.5f\n" % test_history['reconstrution'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    generator.load_weights(model_generator_best_weight_out)
    discriminator.load_weights(model_discriminator_best_weight_out)
    generator.save(model_generator_out)  
    generator.save_weights(model_generator_weight_out, True)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    
    pickle.dump({'train': train_history, 'test': test_history},
      open('acgan-history.pkl', 'wb'))




def DeepSS_1dcon_finetune_variant_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,interval_len,seq_end,batch_size,win_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5,postGAN='None'):
    import numpy as np
    start=30 ## since in the build_discriminator_variant1D, I used kmax30, so the sequence < 30 should be removed
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    Test_data_keys = dict()
    Test_targets_keys = dict()
    feature_num=0; # the number of features for each residue
    
    for key in data_all_dict_padding.keys():
        if key > end: # run first model on 100 at most
          continue
        #print "\n### Loading sequence length :", key
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_labels = trainfeaturedata[:,:,0:3]
        train_labels_convert = np.argmax(train_labels, axis=2) ## need convert to number for gan
        train_feature = trainfeaturedata[:,:,3:]
        feature_num=train_feature.shape[2]/AA_win
        if seq_len in testdata_all_dict_padding:
          testfeaturedata = testdata_all_dict_padding[seq_len]
          #print "Loading test dataset "
        else:
          testfeaturedata = trainfeaturedata
          print "\n\n##Warning: Setting training dataset as testing dataset \n\n"
        
        
        test_labels = testfeaturedata[:,:,0:3]
        test_labels_convert = np.argmax(test_labels, axis=2) ## need convert to number for gan
        test_feature = testfeaturedata[:,:,3:]    
        sequence_length = seq_len
        
        if seq_len in Train_data_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_data_keys[seq_len]=(train_feature)
          
        if seq_len in Train_targets_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_targets_keys[seq_len]=train_labels_convert        
        #processing test data 
        if seq_len in Test_data_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_data_keys[seq_len]=test_feature 
        
        if seq_len in Test_targets_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_targets_keys[seq_len]=test_labels_convert
    
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
        train_labels = featuredata[:,0:3]#(169, 3)
        train_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Trainlist_data_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_data_keys[pdb_name]=train_feature.reshape(1,train_feature.shape[0],train_feature.shape[1])
        
        if pdb_name in Trainlist_targets_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_targets_keys[pdb_name]=train_labels.reshape(1,train_labels.shape[0],train_labels.shape[1])
    
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
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Testlist_data_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_data_keys[pdb_name]=test_feature.reshape(1,test_feature.shape[0],test_feature.shape[1])
        
        if pdb_name in Testlist_targets_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_targets_keys[pdb_name]=test_labels.reshape(1,test_labels.shape[0],test_labels.shape[1])
    
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
        val_labels = featuredata[:,0:3]#(169, 3)
        val_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Vallist_data_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_data_keys[pdb_name]=val_feature.reshape(1,val_feature.shape[0],val_feature.shape[1])
        
        if pdb_name in Vallist_targets_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_targets_keys[pdb_name]=val_labels.reshape(1,val_labels.shape[0],val_labels.shape[1])
    
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    #nb_layers_generator = nb_layers_generator 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = nb_filters # this is for discriminator
    nb_filters_generator = fea_num * AA_win # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    ### Define the model 
    #model_generator_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    #model_generator_best_out= "%s/model-train-generator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    
    #model_generator_best_weight_out= "%s/model-train-generator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    
    #model_generator_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
        
    model_discriminator_postGAN_initial= "%s/model-train-discriminator-postGAN-init-%s.hdf5" % (CV_dir,model_prefix)
    
    if postGAN != 'None':
        if os.path.exists(postGAN):
          print "######## Loading existing postGAN discriminator model ",postGAN;
          discriminator=Sequential()
          discriminator=load_model(postGAN)
        else:
          raise Exception("Failed to find postGAN model (%s) " % postGAN)
    elif os.path.exists(model_discriminator_out):
        print "######## Loading existing discriminator model ",model_discriminator_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_out)
    else:    
        # build the discriminator
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers_discriminator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator_postGAN_variant1D(AA_win,nb_filters,nb_layers_discriminator,win_array,fea_num,n_class)

    if os.path.exists(model_discriminator_weight_out):
    	print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
    	discriminator.load_weights(model_discriminator_weight_out)
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['sparse_categorical_crossentropy'])
    else:
    	print "######## Setting initial discriminator weights";
    	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['sparse_categorical_crossentropy'])
    
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
    
    
    ### train by length   
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        epoch_test_loss = []
        
        for key in data_all_dict_padding.keys():
            if key <start: # run first model on 100 at most
              continue
            if key > end: # run first model on 100 at most
              continue
            #print '### Loading sequence length :', key
            seq_len=key
            
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            test_featuredata_all=Test_data_keys[seq_len]
            test_targets=Test_targets_keys[seq_len]
            #print "Train shape: ",train_featuredata_all.shape, " in outside epoch ", epoch 
            #print "Test shape: ",test_featuredata_all.shape, " in outside epoch ", epoch
                
            
            
            X_train = train_featuredata_all
            y_train = train_targets
            
            X_test = test_featuredata_all
            y_test = test_targets
            
            nb_train, nb_test = X_train.shape[0], X_test.shape[0]
            
            if X_train.shape[0] % batch_size != 0:
              nb_batches = int(X_train.shape[0] / batch_size) + 1
            else:
              nb_batches = int(X_train.shape[0] / batch_size)
            
            #print "\n\n#### Start training GAN: ";
            #print "         X_train: ",X_train.shape;
            #print "         y_train: ",y_train.shape;
            #print "         X_test: ",X_test.shape;
            #print "         y_test: ",y_test.shape;
            #print "         Len: ",seq_len;
            #print "         batch_size: ",batch_size;
            #print "         nb_batches: ",nb_batches;
            progress_bar = Progbar(target=nb_batches)
            for index in range(nb_batches):
                #progress_bar.update(index)
                
                # get a batch of real images
                sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]
                label_batch = label_batch.reshape((label_batch.shape[0],label_batch.shape[1],1))#(20,255,1)
                #print "sample_batch.shape: ",sample_batch.shape
                #print "label_batch.shape: ",label_batch.shape
                
                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(discriminator.train_on_batch(sample_batch, label_batch))
                
            
        print('\nTesting for epoch {}:'.format(epoch + 1))
        # evaluate the testing loss here
        y_test = y_test.reshape((y_test.shape[0],y_test.shape[1],1))#(20,255,1)
        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X_test, y_test, verbose=False)
        
        #print "Average means"
        #print "epoch_gen_loss: ",epoch_gen_loss
        #print "epoch_disc_loss: ",epoch_disc_loss
        #print "epoch_test_loss: ",epoch_test_loss
        #print "reconstruct_gen_loss: ",reconstruct_gen_loss
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        
        # generate an epoch report on performance
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
        
            
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
        print "start evaluating test"
        for i in xrange(len(sequence_file)):
            pdb_name = sequence_file[i].rstrip()
            if pdb_name not in Testlist_data_keys:
                    print 'removing ',pdb_name
                    continue
            test_featuredata_all=Testlist_data_keys[pdb_name]
            test_targets=Testlist_targets_keys[pdb_name]
            result= discriminator.predict([test_featuredata_all])
            #print "test_featuredata_all: ",test_featuredata_all.shape
            #print "result: ",result
            #print "result: ",result.shape
            predict_val=result
            #print "pdb_name: ",pdb_name;
            #print "test_featuredata_all: ",test_featuredata_all.shape
            #print "predict_val: ",predict_val.shape;
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            test_labels_convert = np.argmax(test_targets, axis=2).ravel()
            
            test_acc +=float(sum(test_labels_convert == preds_convert))/len(test_labels_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del test_featuredata_all
            del test_targets
        
        test_acc /= acc_num 
        
        test_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
              myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Trainlist_data_keys:
                    continue
            train_featuredata_all=Trainlist_data_keys[pdb_name]
            train_targets=Trainlist_targets_keys[pdb_name]
            result= discriminator.predict([train_featuredata_all])
            predict_val=result
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            train_targets_convert = np.argmax(train_targets, axis=2).ravel()
            train_acc +=float(sum(train_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del train_featuredata_all
            del train_targets
        
        train_acc /= acc_num
        
        train_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
              myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Vallist_data_keys:
                    continue
            val_featuredata_all=Vallist_data_keys[pdb_name]
            val_targets=Vallist_targets_keys[pdb_name]
            result= discriminator.predict([val_featuredata_all])
            predict_val=result
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            val_targets_convert = np.argmax(val_targets, axis=2).ravel()
            
            val_acc +=float(sum(val_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del val_featuredata_all
            del val_targets
        
        val_acc /= acc_num
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
              myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            #generator.save_weights(model_generator_best_weight_out, True)        
            #generator.save(model_generator_best_out)
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        """
        if epoch % 10== 0 and epoch > 0:
          args_str ="perl "+ lib_dir +"/visualize_training_score_variablegan.pl "  + CV_dir
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          
          summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
          check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
          found = 0
          while (found == 0):
            print "Checking file ",check_file
            time.sleep(15) 
            if os.path.exists(check_file):
              found = 1
          print "Temporary visualization saved to file ",summary_file
          
          image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
          
          args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          found = 0
          while (found == 0):
              print "Checking file ",image_file
              time.sleep(15) 
              if os.path.exists(image_file):
                found = 1
          print "Temporary visualization saved to file ",image_file
        
              
        """           
        print('Training summary:')
        print('-' * 65)
        
        print "discriminator loss (train): %.3f" % train_history['discriminator'][-1] 
        print "discriminator loss (val): %.3f" % test_history['discriminator'][-1]
        
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('Training summary:')
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          myfile.write("discriminator (train): %.3f" % train_history['discriminator'][-1])                             
          myfile.write("discriminator (val): %.3f" % test_history['discriminator'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    discriminator.load_weights(model_discriminator_best_weight_out)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    
    pickle.dump({'train': train_history, 'test': test_history},
      open('acgan-history.pkl', 'wb'))





def DeepSS_1dconv_gan_variant_train_win_filter_layer_opt_V2(data_all_dict_padding,testdata_all_dict_padding,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,interval_len,seq_end,batch_size,win_array,nb_filters,nb_layers_generator,nb_layers_discriminator,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5):
    import numpy as np
    start=30 ## since in the build_discriminator_variant1D, I used kmax30, so the sequence < 30 should be removed
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    Test_data_keys = dict()
    Test_targets_keys = dict()
    feature_num=0; # the number of features for each residue
    
    for key in data_all_dict_padding.keys():
        if key > end: # run first model on 100 at most
          continue
        #print "\n### Loading sequence length :", key
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_labels = trainfeaturedata[:,:,0:3]
        train_labels_convert = np.argmax(train_labels, axis=2) ## need convert to number for gan
        train_feature = trainfeaturedata[:,:,3:]
        feature_num=train_feature.shape[2]/AA_win
        if seq_len in testdata_all_dict_padding:
          testfeaturedata = testdata_all_dict_padding[seq_len]
          #print "Loading test dataset "
        else:
          testfeaturedata = trainfeaturedata
          print "\n\n##Warning: Setting training dataset as testing dataset \n\n"
        
        
        test_labels = testfeaturedata[:,:,0:3]
        test_labels_convert = np.argmax(test_labels, axis=2) ## need convert to number for gan
        test_feature = testfeaturedata[:,:,3:]    
        sequence_length = seq_len
        
        if seq_len in Train_data_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_data_keys[seq_len]=(train_feature)
          
        if seq_len in Train_targets_keys:
          raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
          Train_targets_keys[seq_len]=train_labels_convert        
        #processing test data 
        if seq_len in Test_data_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_data_keys[seq_len]=test_feature 
        
        if seq_len in Test_targets_keys:
          raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
          Test_targets_keys[seq_len]=test_labels_convert
    
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
        train_labels = featuredata[:,0:3]#(169, 3)
        train_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Trainlist_data_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_data_keys[pdb_name]=train_feature.reshape(1,train_feature.shape[0],train_feature.shape[1])
        
        if pdb_name in Trainlist_targets_keys:
          print "Duplicate pdb name %s in Train list " % pdb_name
        else:
          Trainlist_targets_keys[pdb_name]=train_labels.reshape(1,train_labels.shape[0],train_labels.shape[1])
    
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
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Testlist_data_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_data_keys[pdb_name]=test_feature.reshape(1,test_feature.shape[0],test_feature.shape[1])
        
        if pdb_name in Testlist_targets_keys:
          print "Duplicate pdb name %s in Test list " % pdb_name
        else:
          Testlist_targets_keys[pdb_name]=test_labels.reshape(1,test_labels.shape[0],test_labels.shape[1])
    
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
        val_labels = featuredata[:,0:3]#(169, 3)
        val_feature = featuredata[:,3:] #(169, 48)
        if fea_len <start: # run first model on 100 at most
              continue   
        if pdb_name in Vallist_data_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_data_keys[pdb_name]=val_feature.reshape(1,val_feature.shape[0],val_feature.shape[1])
        
        if pdb_name in Vallist_targets_keys:
          print "Duplicate pdb name %s in Val list " % pdb_name
        else:
          Vallist_targets_keys[pdb_name]=val_labels.reshape(1,val_labels.shape[0],val_labels.shape[1])
    
    # batch and latent size taken from the paper
    nb_epochs = epoch_outside
    batch_size = batch_size
    
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    latent_size = latent_size
    adam_lr = adam_lr
    adam_beta_1 = adam_beta_1
    
    ## set network parameter 
    nb_layers_generator = nb_layers_generator 
    nb_layers_discriminator = nb_layers_discriminator
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = nb_filters # this is for discriminator
    nb_filters_generator = fea_num # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    ### Define the model 
    model_generator_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_out= "%s/model-train-generator-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_weight_out= "%s/model-train-generator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s-best.hdf5" % (CV_dir,model_prefix)
    
    model_generator_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    
    if os.path.exists(model_discriminator_out):
        print "######## Loading existing discriminator model ",model_discriminator_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_out, custom_objects={'K_max_pooling1d': K_max_pooling1d})       
    else:    
        # build the discriminator outside since the structure won't change
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers_discriminator;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator_variant1D(nb_filters,nb_layers_discriminator,win_array,AA_win,fea_num)
    
    
    if os.path.exists(model_discriminator_weight_out):
        print "######## Loading existing discriminator weights ",model_discriminator_weight_out;
        discriminator.load_weights(model_discriminator_weight_out)
        discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    else:
        print "######## Setting initial discriminator weights";
        discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    
    print "\n\n#### Summary of discriminator: ";
    print(discriminator.summary())
    
    
    
    # build the generator inside epoch
    
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tAA_window\tEpoch_outside\tAccuracy\n")
    
    GAN_history_out = "%s/GAN_training.history" % (CV_dir)
    chkdirs(GAN_history_out)     
    with open(GAN_history_out, "w") as myfile:
      myfile.write("1D Generative adversarial networks (GANs) for secondary structure prediction\n")
    
    
    ### train by length   
    for epoch in range(0,epoch_outside):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        epoch_test_loss = []
        
        for key in data_all_dict_padding.keys():
            if key <start: # run first model on 100 at most
              continue
            if key > end: # run first model on 100 at most
              continue
            #print '### Loading sequence length :', key
            seq_len=key
            
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            test_featuredata_all=Test_data_keys[seq_len]
            test_targets=Test_targets_keys[seq_len]
            #print "Train shape: ",train_featuredata_all.shape, " in outside epoch ", epoch 
            #print "Test shape: ",test_featuredata_all.shape, " in outside epoch ", epoch
            
            X_train = train_featuredata_all # (n, seq_len, fea_num)
            y_train = train_targets
            
            X_test = test_featuredata_all
            y_test = test_targets
            
            nb_train, nb_test = X_train.shape[0], X_test.shape[0]
            
            if X_train.shape[0] % batch_size != 0:
              nb_batches = int(X_train.shape[0] / batch_size) + 1
            else:
              nb_batches = int(X_train.shape[0] / batch_size)
            
            #print "\n\n#### Start training GAN: ";
            #print "         X_train: ",X_train.shape;
            #print "         y_train: ",y_train.shape;
            #print "         X_test: ",X_test.shape;
            #print "         y_test: ",y_test.shape;
            #print "         Len: ",seq_len;
            #print "         batch_size: ",batch_size;
            #print "         nb_batches: ",nb_batches;
            progress_bar = Progbar(target=nb_batches)
            for index in range(nb_batches):
                #progress_bar.update(index)
                
                # get a batch of real images
                sample_batch = X_train[index * batch_size:(index + 1) * batch_size] ## (batch, seq_len, AA_win * fea_num)
                label_batch = y_train[index * batch_size:(index + 1) * batch_size] # (batch, seq_len, 1)
                #print "sample_batch.shape: ",sample_batch.shape
                #print "label_batch.shape: ",label_batch.shape
                batch_size_indata = batch_size
                if sample_batch.shape[0] < batch_size:
                    batch_size_indata = sample_batch.shape[0]  ## means sample less than batch size
                
                
                #latent = Input(shape=(seq_len,latent_size))
                #ss_class = Input(shape=(seq_len,), dtype='int32')
                
                # get a fake image
                #fake = generator([latent, ss_class])
                
                # here need fix, generate residue by residue and then combine        
                #print "\n\n#### Start initializing generator: ";
                #print "         latent_size: ",latent_size;
                #print "         AA_win: ",AA_win;
                #print "         nb_filters: ",nb_filters_generator;
                #print "         nb_layers: ",nb_layers_generator;
                #print "         win_array: ",win_array;
                #print "         fea_num: ",fea_num;
                #print "         n_class: ",n_class;        
                generator = build_generator_variant1D_V2(latent_size,batch_size_indata,seq_len,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class) #this will generate n * (AA_win * fea_num)
                generator_fake = build_generator_variant1D_V2(latent_size,2*batch_size_indata,seq_len,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class) #this will generate n * (AA_win * fea_num)
                generator_test = build_generator_variant1D_V2(latent_size,nb_test,seq_len,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class) #this will generate n * (AA_win * fea_num)
                generator_fake_test = build_generator_variant1D_V2(latent_size,2*nb_test,seq_len,AA_win,nb_filters_generator,nb_layers_generator,win_array,fea_num,n_class) #this will generate n * (AA_win * fea_num)
                
                   
                if os.path.exists(model_generator_weight_out):
                    #print "######## Loading existing model_generator_weight_out ",model_generator_weight_out;
                    generator.load_weights(model_generator_weight_out)
                    generator_fake.load_weights(model_generator_weight_out)
                    generator_test.load_weights(model_generator_weight_out)
                    generator_fake_test.load_weights(model_generator_weight_out)
                    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_fake.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_test.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_fake_test.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                else:
                    print "######## Setting initial generator weights";
                    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_fake.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_test.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                    generator_fake_test.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy')
                 
                
                #print "\n\n#### Summary of Generator: ";
                #print(generator.summary())
                
                
                latent = Input(shape=(seq_len,latent_size))
                ss_class = Input(shape=(seq_len,), dtype='int32')
        
                # get a fake image
                fake = generator_fake([latent, ss_class])
                fake_test = generator_fake_test([latent, ss_class])

                # we only want to be able to train generation for the combined model
                discriminator.trainable = False
                fake, aux = discriminator(fake)
                fake_test, aux_test = discriminator(fake_test)
                combined = Model(input=[latent, ss_class], output=[fake, aux])
                combined_test = Model(input=[latent, ss_class], output=[fake_test, aux_test])
                #combined.summary()
                combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
                combined_test.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
                
                ### start generate fake feature for full length proteins (this is new strategy using same as fixed-length method, but the batch_size_indata now is batch_size_indata * seq_len)
                # generate a new batch of noise
                noise = np.random.uniform(-1, 1, (batch_size_indata, seq_len, latent_size))   
                # sample some labels from p_c
                sampled_labels = np.random.randint(0, n_class, (batch_size_indata,seq_len)) 
                
                # generate a batch of fake images, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (batch_size, 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_samples = generator.predict_on_batch([noise, sampled_labels.reshape((batch_size_indata, seq_len))])  
                #print "generated_samples: ",generated_samples.shape	
                
                
                X = np.concatenate((sample_batch, generated_samples))
                y = np.array([1] * batch_size_indata + [0] * batch_size_indata)
                
                sampled_labels_new = sampled_labels.reshape(batch_size_indata , seq_len)
                aux_y = np.concatenate((label_batch, sampled_labels_new), axis=0)#(20,255)
                aux_y = aux_y.reshape((aux_y.shape[0],aux_y.shape[1],1))#(20,255,1)
                #aux_y_categorical = (np.arange(aux_y.max()+1) == aux_y[...,None]).astype(int)
                
                #### generate reconstruction error
                reconstruct_samples = generator.predict_on_batch([noise, label_batch.reshape((batch_size_indata, seq_len))])
                reconstruction_error = ((sample_batch - reconstruct_samples) ** 2).mean()
                reconstruct_gen_loss.append(reconstruction_error)
                #print "in: reconstruct_gen_loss: ",reconstruct_gen_loss
                
                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
                #print "in: epoch_disc_loss: ",epoch_disc_loss
                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator
                noise = np.random.uniform(-1, 1, (2 * batch_size_indata, seq_len, latent_size))
                sampled_labels = np.random.randint(0, n_class, (2 * batch_size_indata,seq_len))#(20,255)
                sampled_labels = sampled_labels.reshape((sampled_labels.shape[0],sampled_labels.shape[1],1))#(20,255,1)
                #sampled_labels_categorical = (np.arange(sampled_labels.max()+1) == sampled_labels[...,None]).astype(int)
                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = np.ones(2 * batch_size_indata)
                
                epoch_gen_loss.append(combined.train_on_batch(
                  [noise, sampled_labels.reshape((2 * batch_size_indata, seq_len))], [trick, sampled_labels]))
                #print "in: epoch_gen_loss: ",epoch_gen_loss
                
                # save weights every epoch        
                generator_fake.save_weights(model_generator_weight_out, True)  
                generator_fake.save(model_generator_out)
                      
                discriminator.save_weights(model_discriminator_weight_out, True)
                discriminator.save(model_discriminator_out)
            
            #print('\nTesting for epoch {}:'.format(epoch + 1))
            # evaluate the testing loss here
            
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (nb_test, seq_len,latent_size))
            
            # sample some labels from p_c and generate images from them
            sampled_labels = np.random.randint(0, n_class, (nb_test,seq_len))
            generated_samples = generator_test.predict_on_batch(
              [noise, sampled_labels.reshape((nb_test, seq_len))])
            
            X = np.concatenate((X_test, generated_samples))
            
            
            #### generate reconstruction error using true label
            reconstruct_samples = generator_test.predict_on_batch(
              [noise, y_test.reshape((nb_test, seq_len))])
            reconstruction_test_error = ((X_test - reconstruct_samples) ** 2).mean()
            
            y = np.array([1] * nb_test + [0] * nb_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0)#(20,255)
            aux_y = aux_y.reshape((aux_y.shape[0],aux_y.shape[1],1))#(20,255,1)
            
            #aux_y_categorical = (np.arange(aux_y.max()+1) == aux_y[...,None]).astype(int)
            # see if the discriminator can figure itself out...
            discriminator_test_loss = discriminator.evaluate(
              X, [y, aux_y], verbose=False)
            
            # make new noise
            noise = np.random.uniform(-1, 1, (2 * nb_test, seq_len, latent_size))
            sampled_labels = np.random.randint(0, n_class, (2 * nb_test,seq_len))#(20,255)
            sampled_labels = sampled_labels.reshape((sampled_labels.shape[0],sampled_labels.shape[1],1))#(20,255,1)
            #sampled_labels_categorical = (np.arange(sampled_labels.max()+1) == sampled_labels[...,None]).astype(int)
            
            trick = np.ones(2 * nb_test)
            
            epoch_test_loss.append(combined_test.evaluate(
              [noise, sampled_labels.reshape((2 * nb_test, seq_len))],
              [trick, sampled_labels], verbose=False,batch_size=2 * nb_test))
        #print "Average means"
        #print "epoch_gen_loss: ",epoch_gen_loss
        #print "epoch_disc_loss: ",epoch_disc_loss
        #print "epoch_test_loss: ",epoch_test_loss
        #print "reconstruct_gen_loss: ",reconstruct_gen_loss
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_test_loss = np.mean(np.array(epoch_test_loss), axis=0)
        reconstruction_train_loss = np.mean(np.array(reconstruct_gen_loss), axis=0)
        
        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['reconstrution'].append(reconstruction_train_loss)
        
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        test_history['reconstrution'].append(reconstruction_test_error)
        
        
         # save weights every epoch
        #generator_weigth_out= "%s/params_generator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)
        
        generator.save(model_generator_out)  
        generator.save_weights(model_generator_weight_out, True)
                
        generator_model_out_tmp= "%s/%s/params_generator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(generator_model_out_tmp)      
        generator.save(generator_model_out_tmp)
        
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        
        
        discriminator.save(model_discriminator_out)
        discriminator.save_weights(model_discriminator_weight_out, True)
        
        discriminator_model_out_tmp= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out_tmp)          
        discriminator.save(discriminator_model_out_tmp)
        
        ## how to select the best model and save 
        
            
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
        print "start evaluating test"
        for i in xrange(len(sequence_file)):
            pdb_name = sequence_file[i].rstrip()
            if pdb_name not in Testlist_data_keys:
                    print 'removing ',pdb_name
                    continue
            test_featuredata_all=Testlist_data_keys[pdb_name]
            test_targets=Testlist_targets_keys[pdb_name]
            result= discriminator.predict([test_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            #print "pdb_name: ",pdb_name;
            #print "test_featuredata_all: ",test_featuredata_all.shape
            #print "predict_val: ",predict_val.shape;
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            test_labels_convert = np.argmax(test_targets, axis=2).ravel()
            
            test_acc +=float(sum(test_labels_convert == preds_convert))/len(test_labels_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del test_featuredata_all
            del test_targets
        
        test_acc /= acc_num 
        
        test_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,test_acc)
        with open(test_acc_history_out, "a") as myfile:
              myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Trainlist_data_keys:
                    continue
            train_featuredata_all=Trainlist_data_keys[pdb_name]
            train_targets=Trainlist_targets_keys[pdb_name]
            result= discriminator.predict([train_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            train_targets_convert = np.argmax(train_targets, axis=2).ravel()
            train_acc +=float(sum(train_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del train_featuredata_all
            del train_targets
        
        train_acc /= acc_num
        
        train_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,train_acc)
        with open(train_acc_history_out, "a") as myfile:
              myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
            if pdb_name not in Vallist_data_keys:
                    continue
            val_featuredata_all=Vallist_data_keys[pdb_name]
            val_targets=Vallist_targets_keys[pdb_name]
            result= discriminator.predict([val_featuredata_all])
            predict_val=result[1]
            predict_val_truth=result[0]
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            preds_convert = np.argmax(preds, axis=1)
            val_targets_convert = np.argmax(val_targets, axis=2).ravel()
            
            val_acc +=float(sum(val_targets_convert == preds_convert))/len(train_targets_convert)
            acc_num += 1
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del val_featuredata_all
            del val_targets
        
        val_acc /= acc_num
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\n" % (interval_len,AA_win,epoch,val_acc)
        with open(val_acc_history_out, "a") as myfile:
              myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        
        ##### save the best models, based on mse or acc?
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed        
            generator_fake.save_weights(model_generator_best_weight_out, True)        
            generator_fake.save(model_generator_best_out)
            
            discriminator.save_weights(model_discriminator_best_weight_out, True)
            discriminator.save(model_discriminator_best_out)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        
        if epoch % 10== 0 and epoch > 0:
          args_str ="perl "+ lib_dir +"/visualize_training_score_variablegan.pl "  + CV_dir
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          
          summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
          check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
          found = 0
          while (found == 0):
            print "Checking file ",check_file
            time.sleep(15) 
            if os.path.exists(check_file):
              found = 1
          print "Temporary visualization saved to file ",summary_file
          
          image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
          
          args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
          #print "Running "+ args_str
          args = shlex.split(args_str)
          pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
          
          found = 0
          while (found == 0):
              print "Checking file ",image_file
              time.sleep(15) 
              if os.path.exists(image_file):
                found = 1
          print "Temporary visualization saved to file ",image_file
        
              
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
          'component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                   *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (val)',
                   *test_history['generator'][-1]))
                   
        print(ROW_FMT.format('discriminator (train)',
                   *train_history['discriminator'][-1]))                             
        print(ROW_FMT.format('discriminator (val)',
                   *test_history['discriminator'][-1]))
                   
        print 'Reconstruction Error (train): ', train_history['reconstrution'][-1]
        print 'Reconstruction Error (val): ',test_history['reconstrution'][-1]
        print 'Classification Acc (train): ', train_acc
        print 'Classification Acc (val): ',val_acc
        print 'Classification Acc (test): ',test_acc
        
        with open(GAN_history_out, "a") as myfile:
          myfile.write('\n\nTesting for epoch {}:'.format(epoch + 1))
          myfile.write('\n{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
              'component', *discriminator.metrics_names))
          myfile.write('\n')
          myfile.write('-' * 65)
          myfile.write('\n')
          ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}\n'
          myfile.write(ROW_FMT.format('generator (train)',
                     *train_history['generator'][-1]))
          myfile.write(ROW_FMT.format('generator (test)',
                     *test_history['generator'][-1]))
                     
          myfile.write(ROW_FMT.format('discriminator (train)',
                     *train_history['discriminator'][-1]))                             
          myfile.write(ROW_FMT.format('discriminator (val)',
                     *test_history['discriminator'][-1]))       
          myfile.write("Reconstruction Error (train): %.5f\n" %  train_history['reconstrution'][-1])
          myfile.write("Reconstruction Error (val): %.5f\n" % test_history['reconstrution'][-1])
          myfile.write("Classification Acc (train): %.5f\n" % train_acc)
          myfile.write("Classification Acc (val): %.5f\n" % val_acc)
          myfile.write("Classification Acc (test): %.5f\n" % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    generator.load_weights(model_generator_best_weight_out)
    discriminator.load_weights(model_discriminator_best_weight_out)
    generator.save(model_generator_out)  
    generator.save_weights(model_generator_weight_out, True)
    discriminator.save(model_discriminator_out)
    discriminator.save_weights(model_discriminator_weight_out, True)
    
    pickle.dump({'train': train_history, 'test': test_history},
      open('acgan-history.pkl', 'wb'))




def DeepSS_1dconv_train_win_filter_layer_opt(data_all_dict_padding,testdata_all_dict_padding,train_list,test_list,val_list,CV_dir,feature_dir,model_prefix,epoch_outside,epoch_inside,interval_len,seq_end,win_array,use_bias,hidden_type,nb_filters,nb_layers,opt,lib_dir): #/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/
    start=0
    end=seq_end
    import numpy as np
    Train_data_keys = dict()
    Train_targets_keys = dict()
    Test_data_keys = dict()
    Test_targets_keys = dict()
    
    feature_num=0; # the number of features for each residue
    for key in data_all_dict_padding.keys():
        if key <start: # run first model on 100 at most
            continue
        if key > end: # run first model on 100 at most
            continue
        print '### Loading sequence length :', key
        seq_len=key
        trainfeaturedata = data_all_dict_padding[key]
        train_labels = trainfeaturedata[:,:,0:3]
        train_feature = trainfeaturedata[:,:,3:]
        feature_num=train_feature.shape[2]
        if seq_len in testdata_all_dict_padding:
            testfeaturedata = testdata_all_dict_padding[seq_len]
            #print "Loading test dataset "
        else:
            testfeaturedata = trainfeaturedata
            print "\n\n##Warning: Setting training dataset as testing dataset \n\n"
        
        
        test_labels = testfeaturedata[:,:,0:3]
        test_feature = testfeaturedata[:,:,3:]    
        sequence_length = seq_len
        
        if seq_len in Train_data_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_data_keys[seq_len]=(train_feature)
            
        if seq_len in Train_targets_keys:
            raise Exception("Duplicate seq length %i in Train list, since it has been combined when loading data " % seq_len)
        else:
            Train_targets_keys[seq_len]=train_labels        
        #processing test data 
        if seq_len in Test_data_keys:
            raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
            Test_data_keys[seq_len]=test_feature 
        
        if seq_len in Test_targets_keys:
            raise Exception("Duplicate seq length %i in Test list, since it has been combined when loading data " % seq_len)
        else:
            Test_targets_keys[seq_len]=test_labels
    
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
        train_labels = featuredata[:,0:3]#(169, 3)
        train_feature = featuredata[:,3:] #(169, 48)        
        if pdb_name in Trainlist_data_keys:
            print "Duplicate pdb name %s in Train list " % pdb_name
        else:
            Trainlist_data_keys[pdb_name]=train_feature.reshape(1,train_feature.shape[0],train_feature.shape[1])
        
        if pdb_name in Trainlist_targets_keys:
            print "Duplicate pdb name %s in Train list " % pdb_name
        else:
            Trainlist_targets_keys[pdb_name]=train_labels.reshape(1,train_labels.shape[0],train_labels.shape[1])
    
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
        val_labels = featuredata[:,0:3]#(169, 3)
        val_feature = featuredata[:,3:] #(169, 48)  
        if pdb_name in Vallist_data_keys:
            print "Duplicate pdb name %s in Val list " % pdb_name
        else:
            Vallist_data_keys[pdb_name]=val_feature.reshape(1,val_feature.shape[0],val_feature.shape[1])
        
        if pdb_name in Vallist_targets_keys:
            print "Duplicate pdb name %s in Val list " % pdb_name
        else:
            Vallist_targets_keys[pdb_name]=val_labels.reshape(1,val_labels.shape[0],val_labels.shape[1])
    
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)
    
    if os.path.exists(model_out):
        print "######## Loading existing model ",model_out;
        # load json and create model
        json_file_model = open(model_out, 'r')
        loaded_model_json = json_file_model.read()
        json_file_model.close()
        
        print("######## Loaded model from disk")
        #DeepSS_CNN = model_from_json(loaded_model_json, custom_objects={'remove_1d_padding': remove_1d_padding}) 
        DeepSS_CNN = model_from_json(loaded_model_json)       
    else:
        print "######## Setting initial model";
        ## ktop_node is the length of input proteins
        DeepSS_CNN = DeepCov_SS_with_paras(win_array,feature_num,use_bias,hidden_type,nb_filters,nb_layers,opt)

    if os.path.exists(model_weight_out):
        print "######## Loading existing weights ",model_weight_out;
        DeepSS_CNN.load_weights(model_weight_out)
        DeepSS_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    else:
        print "######## Setting initial weights";
        DeepSS_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
     
 
    train_acc_best = 0 
    test_acc_best = 0 
    
    val_acc_best = 0
    test_acc_history_out = "%s/testing.acc_history" % (CV_dir)
    chkdirs(test_acc_history_out)     
    with open(test_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAccuracy\tLoss\n")
      
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    chkdirs(train_acc_history_out)     
    with open(train_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAccuracy\tLoss\n")
      
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    chkdirs(val_acc_history_out)     
    with open(val_acc_history_out, "w") as myfile:
      myfile.write("Interval_len\tEpoch_outside\tEpoch_inside\tAccuracy\tLoss\n")
    
    for epoch in range(0,epoch_outside):
        print "\n############ Running epoch ", epoch 
        
        for key in data_all_dict_padding.keys():
            if key <start: # run first model on 100 at most
                continue
            if key > end: # run first model on 100 at most
                continue
            print '### Loading sequence length :', key
            seq_len=key
            
            train_featuredata_all=Train_data_keys[seq_len]
            train_targets=Train_targets_keys[seq_len]
            test_featuredata_all=Test_data_keys[seq_len]
            test_targets=Test_targets_keys[seq_len]
            print "Train shape: ",train_featuredata_all.shape, " in outside epoch ", epoch 
            print "Test shape: ",test_featuredata_all.shape, " in outside epoch ", epoch
            DeepSS_CNN.fit([train_featuredata_all], train_targets, batch_size=50,nb_epoch=epoch_inside,  validation_data=([test_featuredata_all], test_targets), verbose=1)
                        
            # serialize model to JSON
            model_json = DeepSS_CNN.to_json()
            print("Saved model to disk")
            with open(model_out, "w") as json_file:
                json_file.write(model_json)
            del train_featuredata_all
            del train_targets
            del test_featuredata_all
            del test_targets
            
            # serialize weights to HDF5
            print("Saved weight to disk") 
            DeepSS_CNN.save_weights(model_weight_out)
        
        #if epoch < epoch_outside*1/3:
        #    continue
        
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
        
        test_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,test_acc,test_loss)
        with open(test_acc_history_out, "a") as myfile:
                    myfile.write(test_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + test_list + " -tag test_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
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
        train_loss=0.0;
        loss_num=0;
        for i in xrange(len(sequence_file)):
            pdb_name = sequence_file[i].rstrip()
            train_featuredata_all=Trainlist_data_keys[pdb_name]
            train_targets=Trainlist_targets_keys[pdb_name]
            score, accuracy = DeepSS_CNN.evaluate([train_featuredata_all], train_targets, batch_size=10, verbose=0)
            train_acc += accuracy
            acc_num += 1
            
            train_loss += score
            loss_num += 1
            
            predict_val= DeepSS_CNN.predict([train_featuredata_all])
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
        
        train_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,train_acc,train_loss)
        with open(train_acc_history_out, "a") as myfile:
                    myfile.write(train_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + train_list + " -tag train_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
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
        val_loss=0.0;
        loss_num=0;
        for i in xrange(len(sequence_file)):
            pdb_name = sequence_file[i].rstrip()
            val_featuredata_all=Vallist_data_keys[pdb_name]
            val_targets=Vallist_targets_keys[pdb_name]
            score, accuracy = DeepSS_CNN.evaluate([val_featuredata_all], val_targets, batch_size=10, verbose=0)
            val_acc += accuracy
            acc_num += 1
            
            val_loss += score
            loss_num += 1
            
            predict_val= DeepSS_CNN.predict([val_featuredata_all])
            targsize=3
            predict_val= predict_val.reshape(predict_val.shape[1],predict_val.shape[2])
            max_vals = np.reshape(np.repeat(predict_val.max(axis=1), targsize), (predict_val.shape[0], targsize))
            #print "".format(predict_val[0], max_vals[0], (predict_val[0] >= max_vals[0]))
            preds = 1 * (predict_val > max_vals - .0001)
            predfile = predir + pdb_name + ".pred";
            probfile = predir + pdb_name + ".prob";
            np.savetxt(predfile, preds, fmt="%d")
            np.savetxt(probfile, predict_val, fmt="%.6f")                        
            del val_featuredata_all
            del val_targets
        
        val_acc /= acc_num 
        val_loss /= loss_num 
        
        val_acc_history_content = "%i\t%i\t%i\t%.4f\t%.4f\n" % (interval_len,epoch,epoch_inside,val_acc,val_loss)
        with open(val_acc_history_out, "a") as myfile:
                    myfile.write(val_acc_history_content)  
        
        args_str ="perl "+ lib_dir +"/evaluation_dnss_prediction.pl -pred "  + predir +  " -out " + dnssdir + " -list " + val_list + " -tag val_list-epoch_" + str(epoch)
        #print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
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
        
        if val_acc >= val_acc_best:
            val_acc_best = val_acc 
            train_acc_best = train_acc
            test_acc_best = test_acc
            score_imed = "Accuracy of Train/Val/Test: %.4f\t%.4f\t%.4f\n" % (train_acc_best,val_acc_best,test_acc_best)
            print "Saved best weight to disk, ", score_imed
            DeepSS_CNN.save_weights(model_weight_out_best)
        print 'The val accuracy is %.5f' % (val_acc) 
        
        
        if epoch % 10== 0 and epoch > 0:
            args_str ="perl "+ lib_dir +"/visualize_training_score.pl "  + CV_dir
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            
            summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
            check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
            found = 0
            while (found == 0):
                print "Checking file ",check_file
                time.sleep(15) 
                if os.path.exists(check_file):
                  found = 1
            print "Temporary visualization saved to file ",summary_file
            
            image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
            
            args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
            #print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            found = 0
            while (found == 0):
                print "Checking file ",image_file
                time.sleep(15) 
                if os.path.exists(image_file):
                  found = 1
            print "Temporary visualization saved to file ",image_file
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    print "Setting and saving best weights"
    DeepSS_CNN.load_weights(model_weight_out_best)
    DeepSS_CNN.save_weights(model_weight_out)
    

