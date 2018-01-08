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
  for($i=5;$i<=20;$i+=1)
  {
    
    $c++;
    print "\n\n###########  processing layers size $i\n";
  
    $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
    print "Generating $runfile\n";
    open(SH,">$runfile") || die "Failed to write $runfile\n";
    
  
    print SH "#!/bin/bash -l\n";
    print SH "#SBATCH -J  ga_lyd-$c\n";
    print SH "#SBATCH -o ga_lyd-$c-%j.out\n";
    print SH "#SBATCH --partition Lewis\n";
    print SH "#SBATCH --nodes=1\n";
    print SH "#SBATCH --ntasks=1         # leave at '1' unless using a MPI code\n";
    print SH "#SBATCH --cpus-per-task=2  # cores per task\n";
    print SH "#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)\n";
    print SH "#SBATCH --time 2-00:00     # days-hours:minutes\n";
    print SH "#SBATCH --qos=normal\n";
    #print SH "#SBATCH --account=general-gpu  # investors will replace this with their account name\n";
    #print SH "#SBATCH --gres gpu:1\n";
    
    
    
    #print SH "## Force Load the Blacklisted Driver\n";
    print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
    print SH "## Activate python virtual environment\n";
    print SH "source /storage/htc/bdm/tools/python_virtualenv/bin/activate\n";
    print SH "module load R/R-3.3.1\n";
    print SH "## Load Needed Modules\n";
    #print SH "module load cuda/cuda-8.0\n";
  
    print SH "feature_dir=$featuredir/features_win${win}_with_atch_no_aa\n";
    print SH "outputdir=$outputdir\n";
  
    print SH "echo \"#################  Training on window $win\"\n";
    print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/GANSS/2_GANSS_CNN_variable/scripts/train_deepcov_gan_ss_variableLen.py 15 10 5 $i  6 50 10 $win \$feature_dir \$outputdir\n";
  
  
    close SH;
  
  }
}

