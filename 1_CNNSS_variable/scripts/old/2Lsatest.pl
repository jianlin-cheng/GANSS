#!/usr/bin/perl
# Script: 	2Lsstest.pl
# Author: 	Matt Spencer
# Made:		10/2/13
# Last Mod:	11/11/13
#
#
# Input:
#
# Dependencies:

use lib '/home/mattcspencer/DNSA/lib';
use strict;
use Time qw(formatted_localtime);
use DN_SApred qw(test_DN newDN timeprint score_probs score_dnss sort_scores shuffle all_testfeatures check_err load_results write_results); 
use Getopt::Long;

my ($help, $replace, $outdir, $wind, $skipx, $iter, $liter, $arch, $pssm, $atch, $seq, $bound, $reduced, $boost, $lpssm, $latch, $lseq, $lwind, $larch, $tag);
my $opt = GetOptions(	'-r!', \$replace,
			'-h!', \$help,
			'-help!', \$help,
			'-out:s', \$outdir,
			'-wind:i', \$wind,
			'-window:i', \$wind,
			'-w:i', \$wind,
			'-iter:i', \$iter,
			'-liter:i', \$liter,
			'-pssm:i', \$pssm,
			'-atch:i', \$atch,
			'-seq:i', \$seq,
			'-bound:i', \$bound,
			'-red!', \$reduced,
			'-x:i', \$skipx,
			'-boost:i', \$boost,
			'-lpssm:i', \$lpssm,
			'-latch:i', \$latch,
			'-lseq:i', \$lseq,
			'-lwind:i', \$lwind,
			'-lw:i', \$lwind,
			'-lwindow:i', \$lwind,
			'-larch:s', \$larch,
			'-tag:s', \$tag,
			'-arch:s', \$arch );

if ($help || !$tag || !$outdir){
	print_help();
	exit;
}

$outdir .= "/" unless ($outdir =~ m/\/$/);

#####################################
$skipx = 0 unless (defined($skipx));
$replace = 0 unless ($replace);
$iter = 5 unless ($iter);
$pssm = 1 unless (defined($pssm));
$atch = 1 unless (defined($atch));
$seq = 0 unless (defined($seq));
$bound = 0 unless (defined($bound));
$reduced = 0 unless ($reduced);
$wind = 17 unless ($wind);
$boost = 1 unless ($boost);
my $target = 1*$boost;
$arch = "500,500,500,200,$target" unless ($arch);
my @archinfo = split(/,/, $arch);
$liter = 5 unless (defined($liter));
$lpssm = 1 unless (defined($lpssm));
$latch = 0 unless (defined($latch));
$lseq = 0 unless (defined($lseq));
$lwind = 17 unless ($lwind);
$larch = "600,600,600,250,3" unless ($larch);
my @larchinfo = split(/,/, $larch);
#####################################

my $trainfile = "../data/lists/dncon-train.lst";
my $testfile = "../data/lists/dncon-test.lst";
my $ssadir = "../data/ssa/";
my $pssmdir = "../data/pssm/";
my $overdir = "/home/mattcspencer/dnssa/data/ss_pred/";
$outdir = $overdir . $outdir;
my $logfile = $outdir . "log.txt";

my $protlist = `cat $testfile`;
my @testlist = split (/\n/, $protlist);

my @features = ($pssm, $atch, $seq, $bound, $skipx);
my @options = (\@features, $wind, $boost, $reduced);
my @lfeatures = ($lpssm, $latch, $lseq, $bound, $skipx);
my @loptions = (\@lfeatures, $lwind, $reduced);
my @dirs = ($ssadir, $pssmdir);

`mkdir $overdir` unless (-d $overdir);
`mkdir $outdir` unless (-d $outdir);

`rm -r $outdir*` if ($replace);

timeprint($logfile, "\n\nPerforming SS train & test using the parameters:\n"
	."\nGlobal Parameters:\n\tTag: $tag\n\tOutdir: $outdir\n\tL1 iteration: $iter\n\tL2 iterations: $liter\n\tReduced: $reduced\n\tReplace: $replace\n\tSkip X: $skipx\n"
	."\nLayer 1 Parameters:\n\tWindow: $wind\n\tPssm: $pssm\n\tAtch: $atch\n\tSeq: $seq\n\tArch: $arch\n\tBoost: $boost\n\tBound: $bound\n"
	."\nLayer 2 Parameters:\n\tWindow: $lwind\n\tPssm: $lpssm\n\tAtch: $latch\n\tSeq: $lseq\n\tArch: $larch\n\tBound: $bound\n");


####################################################
#	Iterating L1 DNs
####################################################

my @Qscores1;
my @Sovscores1;
my @Dirs1;
my @Qscores2;
my @Sovscores2;
my @Dirs2;
for (my $ii=0; $ii<$iter; $ii++){
	timeprint($logfile, "Beginning L1 iteration $ii", 1);

	my $iterdir1 = $outdir . "1iter$ii/";
	`mkdir $iterdir1` unless (-d $iterdir1);
	my $iterdir2 = $outdir . "2iter$ii/";
	`mkdir $iterdir2` unless (-d $iterdir2);

	my $train1file = $iterdir1 . "train.lst";
	my $train2file = $iterdir2 . "train.lst";

	my (@train1, @train2);
	if (-f $train1file && -f $train2file) {
		my $protlist = `cat $train1file`;
		@train1 = split(/\n/, $protlist);

		$protlist = `cat $train2file`;
		@train2 = split(/\n/, $protlist);
	}
	else {
		my $protlist = `cat $trainfile`;
		my @trainlist = split (/\n/, $protlist);
		shuffle (\@trainlist);
		my $mid = int(@trainlist/2);
		@train1 = @trainlist[0..$mid-1];
		@train2 = @trainlist[$mid..$#trainlist];

		open (OUT, ">$train1file") or die "Couldn't open file $train1file\n";
		foreach (@train1){
			print OUT "$_\n";
		}
		close OUT or die "Coudln't close file $train1file\n";

		open (OUT, ">$train2file") or die "Couldn't open file $train2file\n";
		foreach (@train2){
			print OUT "$_\n";
		}
		close OUT or die "Coudln't close file $train2file\n";
	}

	timeprint($logfile, "Preparing L1 iter$ii DN1");
	my ($predir1, $dnssdir1) =
		newDN(	$logfile, $iterdir1, \@train1, \@train2, 
			\@dirs, \@options, \@archinfo, $tag);

	timeprint($logfile, "Preparing L1 iter$ii DN2");
	my ($predir2, $dnssdir2) = 
		newDN(	$logfile, $iterdir2, \@train2, \@train1, 
			\@dirs, \@options, \@archinfo, $tag);

	timeprint ($logfile, "Evaluating predictions of L1 DN1...");
	my $resultfile1 = $iterdir1 . "results.txt";
	my ($AvgQ3, $AvgSov);
	if (-f $resultfile1){
		timeprint ($logfile, "Previously accomplished.");
		($AvgQ3, $AvgSov) = load_results($resultfile1);
	}
	else {
		($AvgQ3, $AvgSov) = score_dnss($dnssdir1, \@train2);
		write_results($resultfile1, $AvgQ3, $AvgSov);
	}
	next if (check_err($AvgQ3, $logfile));

	push (@Dirs1, $iterdir1);
	push (@Qscores1, $AvgQ3);
	push (@Sovscores1, $AvgSov);

	timeprint ($logfile, "Evaluating predictions of L1 DN2...");
	my $resultfile2 = $iterdir2 . "results.txt";
	my ($AvgQ3, $AvgSov);
	if (-f $resultfile2){
		timeprint ($logfile, "Previously accomplished.");
		($AvgQ3, $AvgSov) = load_results($resultfile2);
	}
	else {
		($AvgQ3, $AvgSov) = score_dnss($dnssdir2, \@train1);
		write_results($resultfile2, $AvgQ3, $AvgSov);
	}
	next if (check_err($AvgQ3, $logfile));
	
	push (@Dirs2, $iterdir2);
	push (@Qscores2, $AvgQ3);
	push (@Sovscores2, $AvgSov);
}

####################################################
#	Finalizing L1
####################################################
timeprint ($logfile, "Finalizing first layer...", 1);

my $index = 0;
my ($index, @scoreorder) = sort_scores(\@Qscores1, \@Sovscores1, \@Qscores2, \@Sovscores2);

my $bestdir1 = $outdir . "1iter$index/";
my $bestdir2 = $outdir . "2iter$index/";
timeprint($logfile, "Best L1 trial: $index");
timeprint($logfile, "Scores obtained: ");
foreach (@scoreorder){
	my ($iter, $Q3, $Sov, $Q32, $Sov2) = @{ $_ };
	timeprint($logfile, "Trial: $iter\tQ3: $Q3\tSov: $Sov\tQ3: $Q32\tSov: $Sov2");
}
print "\n\n";

####################################################
#	Preping for L2
####################################################
timeprint ($logfile, "Preping for L2...");

my $joinpredir = $outdir . "L1preds/";
`mkdir $joinpredir` unless (-d $joinpredir);
my $tempdir = $bestdir2 . "pred/";
`cp $tempdir*.pr* $joinpredir`;
$tempdir = $bestdir1 . "pred/";
`cp $tempdir*.pr* $joinpredir`;

my $featdir = $outdir . "testfeatures/";
`mkdir $featdir` unless (-d $featdir);

my $protlist = `cat $trainfile`;
my @trainlist = split (/\n/, $protlist);
my $protlist = `cat $testfile`;
my @testlist = split (/\n/, $protlist);

timeprint($logfile, "Making testfeatures for test list...");

my @errs = all_testfeatures($featdir, \@dirs, \@testlist, \@options);
foreach (@errs){ check_err($_, $logfile, 1); }

timeprint($logfile, "Testing L1 DN1 on test list to obtain prob files...");

my $dnfile = $bestdir1 . "models/ss.model.dat";
$bestdir1 .= "pred/";
my @errs = test_DN ($dnfile, $featdir, $bestdir1, \@testlist, $target);
foreach (@errs){ check_err($_, $logfile, 1); }

timeprint($logfile, "Testing L1 DN2 on test list to obtain prob files...");

my $dnfile = $bestdir2 . "models/ss.model.dat";
$bestdir2 .= "pred/";
my @errs = test_DN ($dnfile, $featdir, $bestdir2, \@testlist, $target);
foreach (@errs){ check_err($_, $logfile, 1); }

push (@dirs, $joinpredir, $bestdir1, $bestdir2);

####################################################
#	Iterating L2 DN
####################################################

my @Qscores;
my @Sovscores;
for (my $ii=0; $ii<$liter; $ii++){
	timeprint($logfile, "Beginning L2 iteration $ii", 1);

	my $literdir = $outdir . "Liter$ii/";
	`mkdir $literdir` unless (-d $literdir);

	my ($predir, $dnssdir) = 
		newDN(	$logfile, $literdir, \@trainlist, \@testlist, 
			\@dirs, \@loptions, \@larchinfo, $tag);

	timeprint ($logfile, "Evaluating predictions...");
	my $resultfile = $literdir . "results.txt";
	my ($AvgQ3, $AvgSov);
	if (-f $resultfile){
		timeprint ($logfile, "Previously accomplished.");
		($AvgQ3, $AvgSov) = load_results($resultfile);
	}
	else {
		($AvgQ3, $AvgSov) = score_dnss($dnssdir, \@testlist);
		write_results($resultfile, $AvgQ3, $AvgSov);
	}
	next if (check_err($AvgQ3, $logfile));

	push (@Qscores, $AvgQ3);
	push (@Sovscores, $AvgSov);
}

my $index = 0;
my ($index, @scoreorder) = sort_scores(\@Qscores, \@Sovscores);
my $bestQ3;
my $bestSov;

my $bestdir = $outdir . "Liter$index/";
timeprint($logfile, "Best L2 trial: $index");
timeprint($logfile, "Scores obtained: ");
foreach (@scoreorder){
	my ($iter, $Q3, $Sov) = @{ $_ };
	$bestQ3 = $Q3 unless ($bestQ3);
	$bestSov = $Sov unless ($bestSov);
	timeprint($logfile, "Trial: $iter\tQ3: $Q3\tSov: $Sov");
}

timeprint($logfile, "Combining L1 and L2 scores...");
$bestdir .= "pred/";
my @predirs = ($bestdir1, $bestdir2, $bestdir);
my ($comQ3, $comSov) = score_probs(\@predirs, \@testlist);
check_err($comQ3, $logfile);

timeprint($logfile, "Using preds from:\n\t$bestdir1\n\t$bestdir2\n\t$bestdir");
timeprint($logfile, "\nFinal:\tQ3: $comQ3\tSov: $comSov");

if ($bestQ3 + $bestSov > $comQ3 + $comSov) {
	timeprint($logfile, "Summing probs was a detriment.\nFinal: Trial $index\tQ3: $bestQ3\tSov: $bestSov\n");
	$bestdir = $outdir . "Liter$index/dnss/";
	print "Using dnss dir $bestdir\n";
	score_dnss($bestdir, \@testlist, $tag);
}
else {	score_probs(\@predirs, \@testlist, $tag); }


sub print_help {
	print "\nHelp Summary for sstest.pl script\n";
	print "Written by Matt Spencer\n";
	print "\nDescription:\n";
	print "This script trains and tests a DN for secondary structure prediction.\n";
	print "\nRequired input:\n";
	print "\t-tag	: Identifying tag - the score file will be named as such.\n";
	print "\t-out	: Subdirectory of data/ss_pred/ to save intermediate files to.\n";
	print "\nOptions:\n";
	print "Global Options:\n";
	print "\t-iter	: Indicate number of L1 iterations.\n";
	print "\t-liter	: Indicate number of L2 iterations.\n";
	print "\t-red	: Indicates to skip gaps during testing.\n";
	print "\t-x	: Indicates code to handle X residues (0,1,2).\n";
	print "\t-r	: Indicates that previously existing files will be replaced.\n";
	print "Layer 1 Options:\n";
	print "\t-wind	: Indicate the window size to use.\n";
	print "\t-pssm  : Include pssm features.\n";
	print "\t-atch  : Include atchley factors.\n";
	print "\t-seq   : Include seq features.\n";
	print "\t-arch 	: Indicate the architecture to use.\n";
	print "\t-bound : Include boundary features.\n";
	print "\t-boost	: Indicate boost window size.\n";
	print "Layer 2 Options:\n";
	print "\t-lwind : Indicate the second layer window size.\n";
	print "\t-lpssm : include pssm features in second layer.\n";
	print "\t-latch : Include atchley factors in second layer.\n";
	print "\t-lseq  : Include seq features in second layer.\n";
	print "\t-larch : Indicate the architecture of the second layer DN.\n";
	print "\t-iter  : Indicate the number of iterations to attempt.\n";
	print "\t-help  : Print this help message.\n";
	print "\n\n";
}
