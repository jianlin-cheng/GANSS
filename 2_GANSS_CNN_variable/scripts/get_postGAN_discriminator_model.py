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

from keras.models import Sequential, Model,load_model
from Model_construct import build_discriminator_postGAN_variant1D
from keras.optimizers import Adam
from Custom_class import K_max_pooling1d

import sys
if len(sys.argv) != 9:
          print 'please input the right parameters'
          sys.exit(1)

model_in=sys.argv[1] #21
AA_win=int(sys.argv[2]) #21
nb_filters=int(sys.argv[3]) #21
nb_layers_discriminator=int(sys.argv[4]) #21
filtsize=sys.argv[5] #21
fea_num=int(sys.argv[6]) #21
n_class=int(sys.argv[7]) #21
model_out=sys.argv[8] #10


filetsize_array = map(int,filtsize.split("_"))

if os.path.exists(model_in):
    print "######## Loading existing discriminator model ",model_in;
    discriminator=Sequential()
    discriminator=load_model(model_in, custom_objects={'K_max_pooling1d': K_max_pooling1d}) 
else:    
    raise Exception("Failed to find model (%s) " % model_in)


print "\n\n#### Summary of discriminator: ";
print(discriminator.summary())

adam_lr = 0.00005
adam_beta_1 = 0.5
print "\n\n############### Constructing post-GAN model";
# build the discriminator
print "\n\n#### Start initializing discriminator: ";
print "         AA_win: ",AA_win;
print "         nb_filters: ",nb_filters;
print "         nb_layers: ",nb_layers_discriminator;
print "         win_array: ",filetsize_array;
print "         fea_num: ",fea_num;
print "         n_class: ",n_class;
discriminator_postGAN = build_discriminator_postGAN_variant1D(AA_win,nb_filters,nb_layers_discriminator,filetsize_array,fea_num,n_class)
discriminator_postGAN.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['sparse_categorical_crossentropy']
    )

print "\n\n#### Summary of postGAN discriminator: ";
print(discriminator_postGAN.summary())


print "\n\n############### Copying the GAN weights to post-GAN model";

#for layer in model.layers:
#    weights = layer.get_weights() # list of numpy arrays
postGAN_layer_ind=0
for ly in range(0,len(discriminator.layers)):
    discriminatorname = discriminator.layers[ly].get_config()['name']
    
    if discriminatorname == 'generation' or discriminatorname == 'k_max_pooling1d_1' or discriminatorname == 'flatten_1':
        continue
    
    discriminatorPostGANname = discriminator_postGAN.layers[postGAN_layer_ind].get_config()['name']
    
    if discriminatorname != discriminatorPostGANname:
      print "!!!!!!!! discriminator layer %i: %s" % (ly,discriminatorname);
      print "!!!!!!!! discriminatorPostGAN  layer %i: %s\n" % (postGAN_layer_ind,discriminatorPostGANname);
      print "!!!!!!!! layer name not equal between two models\n\n";
    
    print "discriminator layer %i: %s" % (ly,discriminatorname);
    print "discriminatorPostGAN  layer %i: %s" % (postGAN_layer_ind,discriminatorPostGANname);
    print "Start copying weight from %s/discriminator to %s/discriminatorPostGAN\n\n" % (discriminatorname,discriminatorPostGANname);
    #print "discriminator layer weight: ",discriminator.layers[ly].get_weights();
    #print "discriminator_postGAN layer weight: ",discriminator_postGAN.layers[postGAN_layer_ind].get_weights();
    discriminator_postGAN.layers[postGAN_layer_ind].set_weights(discriminator.layers[ly].get_weights())
    postGAN_layer_ind+=1

print "Saved postGAN discriminator weight to ", model_out        
discriminator_postGAN.save(model_out)



