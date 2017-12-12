# GANSS
** 1D Generative Adversarial Network for Protein Secondary Structure Prediction



Installation Steps
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
```

**(E) Start training GANSS** 

Note: This is 1d convolutional neural network pretrained by GAN for secondary structure prediction.

(a) generate 15 window feature without atchley and aa

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
THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/scripts/train_deepcov_gan_ss.py 5 2 10 100 1000 15  /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss_gan/results/
```

**(F) Start training DeepCov_SS** 

Note: This is 1d convolutional neural network based secondary structure prediction.

(a) generate 15 window feature without atchley and aa

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
THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/scripts/train_deepcov_ss.py 15  5 5   nadam '6'  100 3  /storage/htc/bdm/jh7x3/GANSS/GANSS_Datasets/features_win15_no_atch_aa/ /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/results/
```

