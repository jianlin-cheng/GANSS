#!/usr/bin/perl -w
use POSIX;

if (@ARGV != 2 ) {
  print "Usage: <input> <output>\n";
  exit;
}


$outputdir = $ARGV[0];
$sbatch_folder = $ARGV[1];

$c=0;
for($i=1;$i<=15;$i+=1)
{
  
  $c++;
  print "\n\n###########  processing interval size $i\n";

  $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";
  

  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  ST1_it-g-$c\n";
  print SH "#SBATCH -o ST1_it-g-$c-%j.out\n";
  print SH "#SBATCH -p Lewis,gpu3\n";
  print SH "#SBATCH -N 1\n";
  print SH "#SBATCH -n 1\n";
  print SH "#SBATCH --mem 30G\n";
  #print SH "#SBATCH --gres gpu:1\n";
  print SH "#SBATCH -t 2-00:00:00\n";
  
  
  #print SH "## Load Needed Modules\n";
  #print SH "module load cuda/cuda-8.0\n";
  
  #print SH "## Force Load the Blacklisted Driver\n";
  #print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
  
  print SH "## Activate python virtual environment\n";
  print SH "source /storage/htc/bdm/tools/python_virtualenv/bin/activate\n";

  print SH "datadir=/storage/htc/bdm/jh7x3/DeepCov_QA_revision/Training_packages_complex_rosetta/SingleDomain_proteins_training/Train_data_20171020\n";
  print SH "outputdir=$outputdir\n";

  print SH "echo \"#################  Training on inter 15\"\n";
  print SH "THEANO_FLAGS=floatX=float32,device=cpu python /storage/htc/bdm/jh7x3/DeepCov_QA_revision/Training_packages_complex_rosetta/SingleDomain_proteins_training/predict_main_iterative_strategy1_20171019_auto_globalonly_withoutRosetta.py $i 5 5   nadam '6' 30 50 50 3  \$datadir \$outputdir\n";


  close SH;


}
