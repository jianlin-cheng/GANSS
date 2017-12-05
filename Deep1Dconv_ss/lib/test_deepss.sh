#!/bin/bash -l
#SBATCH -J  bin-1
#SBATCH -o bin-1-%j.out
#SBATCH -p Lewis
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 80G
#SBATCH -t 2-00:00:00
## Activate python virtual environment
source /group/bdm/tools/keras_virtualenv/bin/activate
echo "#################  Training secondary structure"

module load R/R-3.3.1

feature_dir=/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win1
outputdir=/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/model_train_win1

## Test Theano
THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/train_deepcov_ss.py  15  5 5   nadam '6'  100 3  $feature_dir $outputdir

