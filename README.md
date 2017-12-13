# GANSS
** 1D Generative Adversarial Network for Protein Secondary Structure Prediction



### Installation Steps
--------------------------------------------------------------------------------------

**(A) Download and Unzip GANSS package**  
```
git clone https://github.com/jianlin-cheng/GANSS.git
```
**(B) Configure GANSS package**  
```
cd GANSS
tar -zxvf GANSS_Datasets.tar.gz

cd DNSS_dataset
tar -zxf caspfasta.tar.gz
tar -zxf chains.tar.gz
tar -zxf dssp.tar.gz
tar -zxf fasta.tar.gz
tar -zxf lists.tar.gz
tar -zxf pssm.tar.gz
tar -zxf ssa.tar.gz

perl configure.pl /storage/htc/bdm/jh7x3/GANSS
```
**(C) (Option if already installed) Install Tensorflow, Keras, and h5py and Update keras.json**  

(a) Install Tensorflow: 
```
sudo pip install tensorflow
```
GPU version is NOT needed. If you face issues, refer to the the tensor flow installation guide at https://www.tensorflow.org/install/install_linux.

(b) Install Keras:
```
sudo pip install keras
```

(c) Install the h5py library:  
```
sudo pip install python-h5py
```

(d) Add the entry [“image_dim_ordering": "th”,] to your keras..json file at ~/.keras/keras.json. Note that if you have not tried to run Keras before, you have have to execute the Tensorflow verification step once so that your keras.json file is created. After the update, your keras.json should look like the one below:  
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```
**(D) verify the installation** 

Note: This is 1d convolutional neural network based secondary structure prediction, including  training, prediction and evaluaton.

```
srun -p Gpu -N1 -n10 --gres gpu:1 --mem=100G --pty /bin/bash
cd examples/
./test_deepss.sh
* check /storage/htc/bdm/jh7x3/GANSS/examples/model_train_win15/filter5_layers5_inter15_optnadam_ftsize6/

cd /storage/htc/bdm/jh7x3/GANSS/examples
source /group/bdm/tools/keras_virtualenv/bin/activate
module load cuda/cuda-8.0
python test_discriminator.py
python test_generator.py
python test_generator_variant1D.py
python test_discriminator_variant1D.py
python test_deepcov_model.py
```



### Experimental Steps
--------------------------------------------------------------------------------------

**(A) Start training GANSS** 

Note: This is 1d convolutional neural network pretrained by GAN for secondary structure prediction.

(a) (option if already generated) generate 15 window feature without atchley and aa

```
cd  /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/scripts/
perl P1_generate_features.pl  -out  features_win15_no_atch_aa -wind 15 -atch 0  -seq 0 -nobound
   ** /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/
```

(b) training GANSS on 15 window feature without atchley and aa
```
srun -p Gpu -N1 -n10 --gres gpu:1 --mem=100G --pty /bin/bash
cd /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan
source /group/bdm/tools/keras_virtualenv/bin/activate
module load cuda/cuda-8.0
module load R/R-3.3.1
THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/scripts/train_deepcov_gan_ss.py 5 2 10 100 1000 15  /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/results/
```

**(B) Start training DeepCov_SS** 

Note: This is 1d convolutional neural network based secondary structure prediction.

(a) (option if already generated) generate 15 window feature without atchley and aa

```
cd  /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/scripts/
perl P1_generate_features.pl  -out  features_win15_no_atch_aa -wind 15 -atch 0  -seq 0 -nobound
   ** /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ 
```

(b) training DeepCov_SS on  15 window feature without atchley and aa
```
srun -p Gpu -N1 -n10 --gres gpu:1 --mem=100G --pty /bin/bash
cd /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss
source /group/bdm/tools/keras_virtualenv/bin/activate
module load cuda/cuda-8.0
module load R/R-3.3.1
THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/scripts/train_deepcov_ss.py 15  5 5   nadam '6'  100 3  /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/results/
```


**(C) Start tunning parameter of GANSS** 

Note: start tunning the GANSS network  on win15

(1) Tunning filter size

```
mkdir -p /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_network_filtersize_tune.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_submit_sbatch.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_scripts  0 30 

perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_para_tune_summarize_filter_conv.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/filter_size_tunning_results /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
** /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
```

(2) Tunning generator layer size

```
mkdir -p /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersGen_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersGen_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_network_layersGen_tune.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/GANSS_Datasets/features_win15_no_atch_aa/  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersGen_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersGen_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_submit_sbatch.pl  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersGen_size_tunning_scripts  0 30 

perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_para_tune_summarize_layers_conv_new.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layers_size_tunning_results /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
** /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
```


(3) Tunning discriminator layer size

```
mkdir -p /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersDis_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersDis_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_network_layersDis_tune.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/GANSS_Datasets/features_win15_no_atch_aa/  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersDis_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersDis_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_submit_sbatch.pl  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layersDis_size_tunning_scripts  0 30 

perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_para_tune_summarize_layers_conv_new.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/layers_size_tunning_results /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
** /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
```


(4) Tunning kernel size

```
mkdir -p /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_network_kernelsize_tune_cpu.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/GANSS_Datasets/features_win15_no_atch_aa/  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_results  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_scripts
perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_submit_sbatch.pl  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_scripts  0 30 


perl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/scripts/parameter_tune_scripts/P1_para_tune_summarize_kernel_width_conv_new.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/kernel_size_tunning_results /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
** /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/GANSS/Deep1Dconv_ss_gan/results/Parameter_tunning_win15_no_atch_aa/parameter_tunning_summary
```

