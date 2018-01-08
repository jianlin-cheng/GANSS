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
    print SH "#SBATCH --partition Lewis\n";
    print SH "#SBATCH --nodes=1\n";
    print SH "#SBATCH --ntasks=1         # leave at '1' unless using a MPI code\n";
    print SH "#SBATCH --cpus-per-task=4  # cores per task\n";
    print SH "#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)\n";
    print SH "#SBATCH --time 0-17:00     # days-hours:minutes\n";
    print SH "#SBATCH --qos=normal\n";
    #print SH "#SBATCH --account=general-gpu  # investors will replace this with their account name\n";
    #print SH "#SBATCH --gres gpu:1\n";
    
    
    
    #print SH "## Force Load the Blacklisted Driver\n";
    #print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
    print SH "## Activate python virtual environment\n";
    print SH "source /storage/htc/bdm/tools/python_virtualenv/bin/activate\n";
    print SH "module load R/R-3.3.1\n";
    print SH "## Load Needed Modules\n";
    #print SH "module load cuda/cuda-8.0\n";
  
    print SH "feature_dir=$featuredir/features_win${win}_with_atch_no_aa\n";
    print SH "outputdir=$outputdir\n";
  
  
    ### #########  Evaluation GAN model
    print SH "echo \"#################  Evaluation GAN model on window $win\"\n\n";
    
    #inter15_filters8_layersGen5_layersDis5_batch10_ftsize6_AA_win15
    print SH "CV_dir=\$outputdir/inter15_filters${i}_layersGen5_layersDis5_batch10_ftsize6_AA_win${win}\n\n";
    
    print SH "if [ -d \"\$CV_dir\" ]; then\n";
    print SH "    echo \"#################  Working directory \$CV_dir\"\n";
    print SH "else\n";
    print SH "    echo \"#################  Failed to find working directory \$CV_dir\"\n";
    print SH "    exit\n";
    print SH "fi\n\n";
    
    
    print SH "if [ -f \"\$CV_dir/model-train-discriminator-deepss_1dconv_varigan-best.hdf5\" ]; then\n";
    print SH "    echo \"#################  Found model \$CV_dir/model-train-discriminator-deepss_1dconv_varigan-best.hdf5\"\n";
    print SH "else\n";
    print SH "    echo \"#################  Failed to find model \$CV_dir/model-train-discriminator-deepss_1dconv_varigan-best.hdf5\"\n";
    print SH "    exit\n";
    print SH "fi\n\n";
    
    
    print SH "mkdir \$CV_dir/final_gan_model_evaluation\n";
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/predict_deepcov_ss_gan.py \$CV_dir/model-train-discriminator-deepss_1dconv_varigan-best.hdf5   \$feature_dir   \$CV_dir/final_gan_model_evaluation $win \n";
    print SH "cd \$CV_dir/final_gan_model_evaluation\n";
    print SH "head */*\n";
    print SH "head */* > final_gan_model_evaluation.summary\n\n"; 

    
    ### get post-GAN model
    print SH "echo \"#################  get post-GAN finetuning  on window $win\"\n\n";


    print SH "cd /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/\n";
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python get_postGAN_discriminator_model.py \$CV_dir/model-train-discriminator-deepss_1dconv_varigan-best.hdf5 $win $i  5  6   26  3  \$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5\n";
  
    ## evaluation post-GAN model
    print SH "echo \"#################  Evaluation postGAN model on window $win\"\n\n";
    print SH "if [ -f \"\$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5\" ]; then\n";
    print SH "    echo \"#################  Found postgan model \$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5\"\n";
    print SH "else\n";
    print SH "    echo \"#################  Failed to find postgan model \$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5\"\n";
    print SH "    exit\n";
    print SH "fi\n\n";
    
    
    print SH "mkdir \$CV_dir/final_postgan_model_evaluation\n";
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/predict_deepcov_ss_postgan.py \$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5   \$feature_dir   \$CV_dir/final_postgan_model_evaluation $win\n";
    print SH "cd \$CV_dir/final_gan_model_evaluation\n";
    print SH "head */*\n";
    print SH "head */* > final_gan_model_evaluation.summary\n\n";
    
    #### start finetune the model
    print SH "echo \"#################  Finetune postGAN model on window $win\"\n\n";
    
    print SH "mkdir \$CV_dir/finetune_postgan_model_training\n";
    
    
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/finetune_deepcov_postgan_ss.py  $i  5 5 6 100 1000 $win  \$feature_dir  \$CV_dir/finetune_postgan_model_training \$CV_dir/model-train-discriminator-deepss_1dconv_postgan-best.hdf5\n";



    ## evaluation post-GAN finetune
    print SH "echo \"#################  Evaluation finetune postGAN model on window $win\"\n\n";
    
    print SH "mkdir \$CV_dir/finetune_postgan_model_evaluation\n";
		
   
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/predict_deepcov_ss_postgan.py \$CV_dir/finetune_postgan_model_training/model-train-discriminator-deepss_1dconv_postgan_finetune.hdf5  \$feature_dir  \$CV_dir/finetune_postgan_model_evaluation $win\n";
    print SH "cd \$CV_dir/finetune_postgan_model_evaluation\n";
    print SH "head */*\n";
    print SH "head */* > finetune_postgan_model_evaluation.summary \n";
		
    close SH;
  
  }
}


