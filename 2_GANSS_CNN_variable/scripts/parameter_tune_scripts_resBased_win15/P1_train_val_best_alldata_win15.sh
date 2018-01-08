#!/bin/bash -l
#SBATCH -J  ss_best
#SBATCH -o ss_best-%j.out
#SBATCH -p gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -t 2-00:00:00
module load cuda/cuda-8.0
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
## Activate python virtual environment
source /storage/htc/bdm/tools/python_virtualenv/bin/activate
module load R/R-3.3.1
feature_dir=/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win15
outputdir=/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/Parameter_tunning_win15/Best_parameter_results_train_all 
echo "#################  Training on inter 15"
THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/lib/train_deepcov_ss_all_train.py 7 8 14   nadam '13' 100 3 $feature_dir $outputdir