# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2017

@author: Jie Hou
"""
import os
from Model_construct import build_generator,build_discriminator
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


def DeepSS_1dconv_gan_train_win_filter_layer_opt(data_all_dict,testdata_all_dict,train_list,test_list,val_list,CV_dir,AA_win,feature_dir,model_prefix,epoch_outside,batch_size,win_array,nb_layers,lib_dir,latent_size = 100,adam_lr = 0.00005,adam_beta_1 = 0.5):
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
    nb_layers = nb_layers 
    win_array=win_array
    #AA_win = 15
    AA_win = AA_win # use mnist to test 1d, just to check if works
    #fea_num = 20
    fea_num = feature_num # use mnist to test 1d, just to check if works
    nb_filters = fea_num # use same as number of feature, so that input and output will get same dimension
    n_class = 3
    #n_class = 10 # only for mnist for test    
    
    ### Define the model 
    model_generator_best_out= "%s/model-train-generator-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_out= "%s/model-train-discriminator-%s.hdf5" % (CV_dir,model_prefix)
    
    model_generator_best_weight_out= "%s/model-train-generator-weight-%s.hdf5" % (CV_dir,model_prefix)
    model_discriminator_best_weight_out= "%s/model-train-discriminator-weight-%s.hdf5" % (CV_dir,model_prefix)
    
    if os.path.exists(model_discriminator_best_out):
        print "######## Loading existing discriminator model ",model_discriminator_best_out;
        discriminator=Sequential()
        discriminator=load_model(model_discriminator_best_out)
        
    else:    
        # build the discriminator
        print "\n\n#### Start initializing discriminator: ";
        print "         AA_win: ",AA_win;
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        discriminator = build_discriminator(AA_win,nb_filters,nb_layers,win_array,fea_num,n_class)
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
      
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
        print "         nb_filters: ",nb_filters;
        print "         nb_layers: ",nb_layers;
        print "         win_array: ",win_array;
        print "         fea_num: ",fea_num;
        print "         n_class: ",n_class;
        generator = build_generator(latent_size,AA_win,nb_filters,nb_layers,win_array,fea_num,n_class)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')
    
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
    print "         X_train: ",X_train.shape;
    print "         y_train: ",y_train.shape;
    print "         X_test: ",X_test.shape;
    print "         y_test: ",y_test.shape;
    
    
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
        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        reconstruct_gen_loss = []
        
        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))
            
            # get a batch of real images
            sample_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            
            # sample some labels from p_c
            sampled_labels = np.random.randint(0, n_class, batch_size)
            
            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_samples = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            
            X = np.concatenate((sample_batch, generated_samples))
            y = np.array([1] * batch_size + [0] * batch_size)
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
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, n_class, 2 * batch_size)
            
            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)
            
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
        #generator.save_weights(generator_weigth_out, True)
        generator_model_out= "%s/%s/params_generator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(generator_model_out)      
        generator.save(generator_model_out)
        
        
        #discriminator_weight_out= "%s/params_discriminator_weight_epoch_%d_%s.hdf5" % (CV_dir,epoch,model_prefix)         
        #discriminator.save_weights(discriminator_weight_out, True)
        discriminator_model_out= "%s/%s/params_discriminator_model_epoch_%d_%s.hdf5" % (CV_dir,'epoch_models',epoch,model_prefix)
        chkdirs(discriminator_model_out)          
        discriminator.save(discriminator_model_out)
        
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
        print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/test_list-epoch_'+str(epoch) + '.score'
        
        found = 0
        while (found == 0):
            print "Checking file ",scorefile
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
        print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/train_list-epoch_'+str(epoch) + '.score'
        
        found = 0
        while (found == 0):
            print "Checking file ",scorefile
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
        print "Running "+ args_str
        args = shlex.split(args_str)
        pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
        
        scorefile=dnssdir+'/val_list-epoch_'+str(epoch) + '.score'
        
        found = 0
        while (found == 0):
            print "Checking file ",scorefile
            time.sleep(5) 
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
            args_str ="perl "+ lib_dir +"/visualize_training_score.pl "  + CV_dir
            print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            
            summary_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary'
            check_file = CV_dir + '/train_val_test.loss_q3_sov_history_summary.done'
            found = 0
            while (found == 0):
                print "Checking file ",check_file
                time.sleep(5) 
                if os.path.exists(check_file):
                  found = 1
            print "Temporary visualization saved to file ",summary_file
            
            image_file = CV_dir + '/train_val_test_loss_q3_sov_history_summary.jpeg'
            
            args_str ="Rscript "+ lib_dir +"/visualize_training_score.R "  + summary_file + "  " + image_file
            print "Running "+ args_str
            args = shlex.split(args_str)
            pipe = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            found = 0
            while (found == 0):
                print "Checking file ",image_file
                time.sleep(5) 
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
          myfile.write('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                  'component', *discriminator.metrics_names))
          myfile.write('-' * 65)
          ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
          myfile.write(ROW_FMT.format('generator (train)',
                               *train_history['generator'][-1]))
          myfile.write(ROW_FMT.format('generator (test)',
                               *test_history['generator'][-1]))
                               
          myfile.write(ROW_FMT.format('discriminator (train)',
                               *train_history['discriminator'][-1]))                             
          myfile.write(ROW_FMT.format('discriminator (val)',
                               *test_history['discriminator'][-1]))       
          myfile.write("Reconstruction Error (train): %.5f " %  train_history['reconstrution'][-1])
          myfile.write("Reconstruction Error (val): %.5f " % test_history['reconstrution'][-1])
          myfile.write("Classification Acc (train): %.5f " % train_acc)
          myfile.write("Classification Acc (val): %.5f " % val_acc)
          myfile.write("Classification Acc (test): %.5f " % test_acc)
    
    #print "Training finished, best training acc = ",train_acc_best
    print "Training finished, best testing acc = ",test_acc_best
    print "Training finished, best validation acc = ",val_acc_best
    print "Training finished, best training acc = ",train_acc_best
    
    pickle.dump({'train': train_history, 'test': test_history},
        open('acgan-history.pkl', 'wb')) 