#!/usr/bin/perl -w
use POSIX;

if (@ARGV != 3 ) {
  print "Usage: <input> <output>\n";
  exit;
}


$featuredir = $ARGV[0];
$outputdir = $ARGV[1];
$sbatch_folder = $ARGV[2];

$c=0;


for($win=1;$win<=19;$win+=2)
{
  for($i=5;$i<=50;$i+=1)
  {
    
    $c++;
    print "\n\n###########  processing filter size $i\n";
  
    $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
    print "Generating $runfile\n";
    open(SH,">$runfile") || die "Failed to write $runfile\n";
    
  
    print SH "#!/bin/bash -l\n";
    print SH "#SBATCH -J  ga2_ft-$c\n";
    print SH "#SBATCH -o ga2_ft-$c-%j.out\n";
    print SH "#SBATCH --partition gpu3\n";
    print SH "#SBATCH --nodes=1\n";
    print SH "#SBATCH --ntasks=1         # leave at '1' unless using a MPI code\n";
    print SH "#SBATCH --cpus-per-task=1  # cores per task\n";
    print SH "#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)\n";
    print SH "#SBATCH --time 2-00:00     # days-hours:minutes\n";
    print SH "#SBATCH --qos=normal\n";
    print SH "#SBATCH --account=general-gpu  # investors will replace this with their account name\n";
    print SH "#SBATCH --gres gpu:1\n";
    
    
    
    #print SH "## Force Load the Blacklisted Driver\n";
    print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
    print SH "## Activate python virtual environment\n";
    print SH "source /storage/htc/bdm/tools/python_virtualenv/bin/activate\n";
    print SH "module load R/R-3.3.1\n";
    print SH "## Load Needed Modules\n";
    print SH "module load cuda/cuda-8.0\n";
  
    print SH "feature_dir=$featuredir/features_win${win}_with_atch_no_aa\n";
    print SH "outputdir=$outputdir\n";
  
  
    ### #########  Evaluation GAN model
    print SH "echo \"#################  Evaluation GAN model on window $win\"\n\n";
    
    print SH "CV_dir=\$outputdir/filters${i}_layersGen5_layersDis5_batch1000_ftsize6_AA_win${win}\n\n";
    
    print SH "if [ -d \"\$CV_dir\" ]; then\n";
    print SH "    echo \"#################  Working directory \$CV_dir\"\n";
    print SH "else\n";
    print SH "    echo \"#################  Failed to find working directory \$CV_dir\"\n";
    print SH "    exit\n";
    print SH "fi\n\n";
    

    
    #### start train the model without pretrain
    print SH "echo \"#################  start train the model without pretrain on window $win\"\n\n";
    
    print SH "mkdir \$CV_dir/finetune_postgan_model_withoutPretraining\n";
    
    
    print SH "THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/jh7x3/GANSS/5_GANSS_CNN_window/scripts/finetune_deepcov_postgan_ss_withoutPretrain.py  $i  5 5 6 100 1000 $win  \$feature_dir \$CV_dir/finetune_postgan_model_withoutPretraining\n";


    ## evaluation post-GAN finetune
    print SH "echo \"#################  Evaluation model on window $win\"\n\n";
    
    print SH "mkdir \$CV_dir/finetune_postgan_model_withoutPretraining_evaluation\n";
   
    print SH "THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/jh7x3/GANSS/5_GANSS_CNN_window/scripts/predict_deepcov_ss_postgan.py \$CV_dir/finetune_postgan_model_withoutPretraining/model-train-discriminator-deepss_1dconv_postgan_finetune_nopretrain.hdf5  \$feature_dir  \$CV_dir/finetune_postgan_model_withoutPretraining_evaluation $win\n";
    print SH "cd \$CV_dir/finetune_postgan_model_withoutPretraining_evaluation\n";
    print SH "head */*\n";
    print SH "head */* > finetune_postgan_model_withoutPretraining_evaluation.summary\n";
		
    close SH;
  
  }
}


