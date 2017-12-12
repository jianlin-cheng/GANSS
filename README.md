# GANSS
** Generative Adversarial Network for Protein Secondary Structure Prediction



Installation Steps
--------------------------------------------------------------------------------------

**(A) Download and Unzip GANSS package**  

git clone https://github.com/jianlin-cheng/GANSS.git

**(B) Configure GANSS package**  
cd GANSS
tar -zxvf GANSS_Datasets.tar.gz


**(C) Install Tensorflow, Keras, and h5py and Update keras.json**  

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

**(D) Start training GANSS** 
