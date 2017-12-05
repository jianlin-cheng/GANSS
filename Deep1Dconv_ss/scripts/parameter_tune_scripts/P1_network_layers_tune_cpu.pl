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
for($i=5;$i<=20;$i+=1)
{
  
  $c++;
  print "\n\n###########  processing layers size $i\n";

  $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";
  

  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  ss_ly-$c\n";
  print SH "#SBATCH -o ss_ly-$c-%j.out\n";
  print SH "#SBATCH -p Lewis\n";
  print SH "#SBATCH -N 1\n";
  print SH "#SBATCH -n 1\n";
  print SH "#SBATCH --mem 20G\n";
  #print SH "#SBATCH --gres gpu:1\n";
  print SH "#SBATCH -t 2-00:00:00\n";
  
  
  #print SH "## Load Needed Modules\n";
  #print SH "module load cuda/cuda-8.0\n";
  
  #print SH "## Force Load the Blacklisted Driver\n";
  #print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
  #print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
  
  print SH "## Activate python virtual environment\n";
  print SH "source /storage/htc/bdm/tools/python_virtualenv/bin/activate\n";
  print SH "module load R/R-3.3.1\n";


  print SH "feature_dir=$featuredir\n";
  print SH "outputdir=$outputdir\n";

  print SH "echo \"#################  Training on inter 15\"\n";
  print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/lib/train_deepcov_ss.py  7 5  $i nadam '6' 100 3 \$feature_dir \$outputdir\n";
  
  close SH;


}
