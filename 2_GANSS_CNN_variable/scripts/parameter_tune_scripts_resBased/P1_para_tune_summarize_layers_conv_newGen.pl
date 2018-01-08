#!/usr/bin/perl -w
use POSIX;

if (@ARGV != 2) {
  print "Usage: <input> <output>\n";
  exit;
}

$result_dir = $ARGV[0]; #/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/Parameter_tunning_win1/filter_size_tunning_results
$outputdir = $ARGV[1];#

open(OUT,">$outputdir/train_data_layerGen_summary.txt") || die "Failed to open file $outputdir/train_data_layerGen_summary.txt\n";
open(OUT2,">$outputdir/validation_layerGen_data_summary.txt") || die "Failed to open file $outputdir/validation_layerGen_data_summary.txt\n";




print OUT "Parameter\tSetting\tMetric\tScore\n";
print OUT2 "Parameter\tSetting\tMetric\tScore\n";

opendir(DIR,"$result_dir") || die "Failed to open directory $result_dir\n";
@subdirs = readdir(DIR);
closedir(DIR);

foreach $dir (sort @subdirs)
{
  chomp $dir;
  if($dir eq '.' or $dir eq '..')
  {
    next;
  }
  $para = substr($dir,index($dir,'_layersGen')+length('_layersGen'),index($dir,'_layersDis')-index($dir,'_layersGen')-length('_layersGen'));
  print "Processing layer size $para\n";
  $train_results = "$result_dir/$dir/train_val_test.loss_q3_sov_history_summary";
  
  if(!(-e $train_results))
  {
    next;
  } 
  open(IN,$train_results) || die "Failed to find $train_results\n";
  @tmp = <IN>;
  close IN;
  
  $q3_train=0;
  $sov_train=0;
  $loss_train=0;
  $q3_val=0;
  $sov_val=0;
  $loss_val=0;
  foreach $line (@tmp)
  {
    chomp $line;
    @array = split(/\t/,$line);
    $dataset = $array[0];
    $epoch = $array[2];
    $metric = $array[3];
    $score = $array[4];
    if($dataset eq 'Validation' and $metric eq 'Q3')
    {
      $q3_val = $score;
    }
    
    if($dataset eq 'Validation' and $metric eq 'SOV')
    {
      $sov_val = $score;
    }
    
    if($dataset eq 'Validation' and $metric eq 'Recon_Err')
    {
      $loss_val = $score;
    }
    
    
    if($dataset eq 'Train' and $metric eq 'Q3')
    {
      $q3_train = $score;
    }
    
    if($dataset eq 'Train' and $metric eq 'SOV')
    {
      $sov_train = $score;
    }
    
    if($dataset eq 'Train' and $metric eq 'Recon_Err')
    {
      $loss_train = $score;
    }
    
                    
  }
  
  print OUT "LayerGen\t$para\tQ3\t$q3_train\n";
  print OUT "LayerGen\t$para\tSOV\t$sov_train\n";
  print OUT "LayerGen\t$para\tRecon_Err\t$loss_train\n";
  
  
  print OUT2 "LayerGen\t$para\tQ3\t$q3_val\n";
  print OUT2 "LayerGen\t$para\tSOV\t$sov_val\n";
  print OUT2 "LayerGen\t$para\tRecon_Err\t$loss_val\n";
  
  
}

close OUT;
close OUT2;
