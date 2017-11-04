#!/usr/bin/perl
# Script: 	P1_generate_features.pl
# Author: 	Jie Hou
# Made:		11/03/17
#
#
# Input:
#
# Dependencies:

use lib "../lib";
use strict;
use Time qw(formatted_localtime);
use DN_SSpred qw(generate_feature_for_convolution timeprint sort_scores score_dnss check_err); 
use Getopt::Long;

my ($help, $replace, $outdir, $wind, $iters, $bound, $reduced, $boost, $skipx, $arch, $pssm, $atch, $seq, $tag);
my $opt = GetOptions(	'-r!', \$replace,
			'-h!', \$help,
			'-help!', \$help,
			'-out:s', \$outdir,
			'-wind:i', \$wind,
			'-window:i', \$wind,
			'-w:i', \$wind,
			'-bound!', \$bound,
			'-red!', \$reduced,
			'-x:i', \$skipx,
			'-boost:i', \$boost,
			'-iter:i', \$iters,
			'-pssm:i', \$pssm,
			'-atch:i', \$atch,
			'-seq:i', \$seq,
			'-tag:s', \$tag,
			'-arch:s', \$arch );

if ($help || !$outdir){
	print_help();
	exit;
}

$outdir .= "/" unless ($outdir =~ m/\/$/);

#####################################
$iters = 1 unless ($iters);
$pssm = 1 unless (defined($pssm));
$atch = 1 unless (defined($atch));
$seq = 1 unless (defined($seq)); ## DNSS exclude this, but I add here first
$wind = 7 unless ($wind);
$boost = 1 unless ($boost);
$skipx = 0 unless ($skipx);
$bound = 1 unless (defined($bound));
$reduced = 0 unless ($reduced);
my $target = 3*$boost;
$arch = "X,X,$target" unless ($arch);
my @archinfo = split(/,/, $arch);
#my $lwind = 17;
#my $larch = "600,600,600,250,3";
#####################################

my $trainfile = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/dncon-train.lst";
my $testfile = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/dncon-test.lst";
my $ssadir = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/ssa/";
my $pssmdir = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/pssm/";
my $overdir = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/"; ## this is jie's working directory
$outdir = $overdir . $outdir;
my $logfile = $outdir . "log.txt";

my @features = ($pssm, $atch, $seq, $bound, $skipx);
my @options = (\@features, $wind, $boost, $reduced);
my @dirs = ($ssadir, $pssmdir);

`mkdir $overdir` unless (-d $overdir);
`mkdir $outdir` unless (-d $outdir);

timeprint($logfile, "Performing SS train & test using the parameters:\nTag: $tag\nOutdir: $outdir\nIterations: $iters\nReduced: $reduced\nReplace: $replace\nSkip X: $skipx\n\nWindow: $wind\nPssm: $pssm\nAtch: $atch\nSeq: $seq\nArch: $arch\nBoost: $boost\nBound: $bound\n");

`rm -r $outdir/*` if ($replace);


my $protlist = `cat $trainfile`;
my @trainlist = split (/\n/, $protlist);
my $protlist = `cat $testfile`;
my @testlist = split (/\n/, $protlist);

generate_feature_for_convolution(	$logfile, $outdir, \@trainlist, \@testlist,	\@dirs, \@options);


################################################################
################################################################

sub print_help {
	print "\nHelp Summary for P1_generate_features.pl script\n";
	print "Written by Jie Hou\n";
	print "Made:		11/03/17\n";
	print "\nDescription:\n";
	print "This script generates training and testing features for secondary structure prediction.\n";
	print "\nRequired input:\n";
	print "\t-out	: Subdirectory of Deep1Dconv_ss/features_win1/ to save intermediate files to.\n";
	print "\t-wind	: Indicate the window size to use.\n";
	print "\nOptions:\n";
	print "\t-r	: Indicates that previously existing files will be replaced.\n";
	print "\t-x	: Deal with X res (0: keep, 1: skip lines, 2: skip windows)\n";
	print "\t-red	: Only predict non-gap residues.\n";
	print "\t-boost	: Indicate the size of boost window.\n";
	print "\t-pssm	: Include pssm features.\n";
	print "\t-atch	: Include atchley factors.\n";
	print "\t-seq	: Include seq features.\n";
	print "\t-bound	: Include boundary features.\n";
	print "\t-arch	: Indicate the architecture to use.\n";
	print "\t-iter	: Indicate the number of iterations to attempt.\n";
	print "\t-help	: Print this help message.\n";
	print "\n\n";
}
