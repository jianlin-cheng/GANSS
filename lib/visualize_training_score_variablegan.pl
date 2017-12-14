#!/usr/bin/env perl

#### written by Jie Hou to plot training figures
### made  2017/11/04

# perl /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/visualize_training_score.pl /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/model_train_win1/filter5_layers5_optnadam_ftsize6

$num = @ARGV;

if($num != 1)
{
	die "The number of parameter is not correct!\n";
}

$working_dir = $ARGV[0]; #


$bin_size=0;
############################  summarize the history information


$train_history = "$working_dir/training.acc_history";
$test_history = "$working_dir/testing.acc_history";
$val_history = "$working_dir/validation.acc_history";
$gan_history = "$working_dir/GAN_training.history";


open(OUT, ">$working_dir/train_val_test.loss_q3_sov_history_summary") or print ("CANNOT open $working_dir/train_val_test.loss_q3_sov_history_summary\n");


$interval_global = 0;
############################  summarize the accuracy information
print OUT "Data\tAA_win\tEpoch\tMetric\tScore\n";
open(IN, "$train_history") or print ("CANNOT open $train_history\n");
while(<IN>)
{
	$line=$_;
	chomp $line;
  if(index($line,'Epoch') > 0)
  {
    next;
  }
  @tmp = split(/\t/,$line);
  
  $interval = $tmp[0];
  $epoch = $tmp[2];
  $acc = $tmp[3];
  $bin_size = $interval;
  print OUT "Train\t$interval\t$epoch\tAccuracy\t$acc\n";
  $interval_global=  $interval;
}
close IN;

open(IN, "$test_history") or print ("CANNOT open $test_history\n");
while(<IN>)
{
	$line=$_;
	chomp $line;
  if(index($line,'Epoch') > 0)
  {
    next;
  }
  @tmp = split(/\t/,$line);
  
  $interval = $tmp[0];
  $epoch = $tmp[2];
  $acc = $tmp[3];
  
  print OUT "Test\t$interval\t$epoch\tAccuracy\t$acc\n";
}
close IN;


open(IN, "$val_history") or print ("CANNOT open $val_history\n");
while(<IN>)
{
	$line=$_;
	chomp $line;
  if(index($line,'Epoch') > 0)
  {
    next;
  }
  @tmp = split(/\t/,$line);
  
  $interval = $tmp[0];
  $epoch = $tmp[2];
  $acc = $tmp[3];
  
  print OUT "Validation\t$interval\t$epoch\tAccuracy\t$acc\n";
}
close IN;


############################  summarize the reconstruction error information
open(IN, "$gan_history") or print ("CANNOT open $gan_history\n");
$epoch_record_train=0;
$epoch_record_val=0;
while(<IN>)
{
	$line=$_; #Reconstruction Error (train):  0.675330660651
	chomp $line;
  if(index($line,'Reconstruction Error ') <  0)
  {
    next;
  }
  
  
  if(index($line,'train') > 0)
  {
    $epoch_record_train++;
    $train_error = substr($line,index($line,':')+1 );
    $train_error =~ s/^\s+|\s+$//g;
    print OUT "Train\t$interval_global\t$epoch_record_train\tRecon_Err\t$train_error\n";
  }
  if(index($line,'val') > 0)
  {
    $epoch_record_val++;
    $val_error = substr($line,index($line,':')+1 );
    $val_error =~ s/^\s+|\s+$//g;
    print OUT "Validation\t$interval_global\t$epoch_record_val\tRecon_Err\t$val_error\n";
  }
}
close IN;

############################  summarize the Q3/SOV information


$train_history_dir = "$working_dir/train_prediction_q3_sov_log_loss";
$test_history_dir = "$working_dir/test_prediction_q3_sov_log_loss";
$val_history_dir = "$working_dir/val_prediction_q3_sov_log_loss";

## train 
for($i=0;$i<200;$i++)
{
  $scorefile = "$train_history_dir/train_list-epoch_$i.score";
  if(!(-e $scorefile))
  {
    next;
  }
  open(IN, "$scorefile") or print ("CANNOT open $scorefile\n");
  $q3_avg = 0;
  $sov_avg = 0;
  while(<IN>)
  {
  	$line=$_;
  	chomp $line;
    if(index($line,'#Average Q3 score') >= 0)
    {
       $q3_avg = substr($line,index($line,':')+1);
       $q3_avg =~ s/^\s+|\s+$//g
    }
    if(index($line,'#Average Sov score') >= 0)
    {
       $sov_avg = substr($line,index($line,':')+1);
       $sov_avg =~ s/^\s+|\s+$//g
    }
  }
  close IN;
  print OUT "Train\t$bin_size\t$i\tQ3\t$q3_avg\n";
  print OUT "Train\t$bin_size\t$i\tSOV\t$sov_avg\n";
}



## test 
for($i=0;$i<200;$i++)
{
  $scorefile = "$test_history_dir/test_list-epoch_$i.score";
  if(!(-e $scorefile))
  {
    next;
  }
  open(IN, "$scorefile") or print ("CANNOT open $scorefile\n");
  $q3_avg = 0;
  $sov_avg = 0;
  while(<IN>)
  {
  	$line=$_;
  	chomp $line;
    if(index($line,'#Average Q3 score') >= 0)
    {
       $q3_avg = substr($line,index($line,':')+1);
       $q3_avg =~ s/^\s+|\s+$//g
    }
    if(index($line,'#Average Sov score') >= 0)
    {
       $sov_avg = substr($line,index($line,':')+1);
       $sov_avg =~ s/^\s+|\s+$//g
    }
  }
  print OUT "Test\t$bin_size\t$i\tQ3\t$q3_avg\n";
  print OUT "Test\t$bin_size\t$i\tSOV\t$sov_avg\n";
  close IN;
}




## validation 
for($i=0;$i<200;$i++)
{
  $scorefile = "$val_history_dir/val_list-epoch_$i.score";
  if(!(-e $scorefile))
  {
    next;
  }
  open(IN, "$scorefile") or print ("CANNOT open $scorefile\n");
  $q3_avg = 0;
  $sov_avg = 0;
  while(<IN>)
  {
  	$line=$_;
  	chomp $line;
    if(index($line,'#Average Q3 score') >= 0)
    {
       $q3_avg = substr($line,index($line,':')+1);
       $q3_avg =~ s/^\s+|\s+$//g
    }
    if(index($line,'#Average Sov score') >= 0)
    {
       $sov_avg = substr($line,index($line,':')+1);
       $sov_avg =~ s/^\s+|\s+$//g
    }
  }
  print OUT "Validation\t$bin_size\t$i\tQ3\t$q3_avg\n";
  print OUT "Validation\t$bin_size\t$i\tSOV\t$sov_avg\n";
  close IN;
}

close OUT;


`cp $working_dir/train_val_test.loss_q3_sov_history_summary $working_dir/train_val_test.loss_q3_sov_history_summary.done`;

##### start plot 
#print " Running <Rscript /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/visualize_training_score.R  $working_dir/train_val_test.loss_q3_sov_history_summary $working_dir/train_val_test_loss_q3_sov_history_summary.jpeg>\n\n";

#system("Rscript /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/visualize_training_score.R  $working_dir/train_val_test.loss_q3_sov_history_summary  $working_dir/train_val_test_loss_q3_sov_history_summary.jpeg");




