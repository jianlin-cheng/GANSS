#!/usr/bin/env perl

use warnings;

use List::Util qw/shuffle/;



$num = @ARGV;

if($num != 2)
{
	die "The number of parameter is not correct!\n";
}

$inputfile = $ARGV[0]; #
$outfolder = $ARGV[1]; #


open(IN, "$inputfile") or print ("CANNOT open $inputfile\n");
open(TRAIN, ">$outfolder/dncov_training.list") or print ("CANNOT open $outfolder/dncov_training.list\n");
open(VAL, ">$outfolder/dncov_validation.list") or print ("CANNOT open $outfolder/dncov_validation.list\n");

%target_idlist=();
while(<IN>)
{
	$target=$_;
	chomp $target;
	$target_idlist{$target}=1;
}
close IN;

@targets = keys %target_idlist;

my @targets_sorted = shuffle @targets;


### generate training targets  80%
%train_targets=();
for($i=0;$i<@targets_sorted*0.9;$i++)
{
	$train_targets{$targets_sorted[$i]}=1;
}


	

open(IN, "$inputfile") or print ("CANNOT open $inputfile\n");
%target_idlist=();
while(<IN>)
{
	$target=$_;
	chomp $target;
	if(exists($train_targets{$target}))
	{
		print TRAIN "$target\n";
	}else{
		print VAL "$target\n";
	}
}
close IN;	
close TRAIN;	
close VAL;