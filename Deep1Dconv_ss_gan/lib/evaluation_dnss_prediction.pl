#!/usr/bin/perl
# Script: 	sstest.pl
# Author: 	Jie Hou
# Made:		9/25/13
# Last Mod:	11/4/17
#
# perl /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/evaluation_dnss_prediction.pl -pred  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/model_train_win1/filter5_layers5_optnadam_ftsize6/test_prediction  -out /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/model_train_win1/filter5_layers5_optnadam_ftsize6/test_prediction_dnss -list /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/adj_dncon-test.lst -tag "test-list"
# Input:
#
# Dependencies:

use lib "/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/";
use strict;
use Time qw(formatted_localtime);
use DN_SSpred qw(newDN timeprint sort_scores score_dnss check_err all_dnss_files); 
use Getopt::Long;

my ($help, $dnssdir, $predir, $testfile, $tag);
my $opt = GetOptions('-h!', \$help,
			'-help!', \$help,
			'-pred:s', \$predir,
			'-out:s', \$dnssdir,
			'-list:s', \$testfile,
			'-tag:s', \$tag);

if ($help || !$tag || !$predir){
	print_help();
	exit;
}

$dnssdir .= "/" unless ($dnssdir =~ m/\/$/);

my $logfile = $dnssdir . "log.txt";
#####################################

my $pssmdir = "/storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/pssm/";

`mkdir $dnssdir` unless (-d $dnssdir);

my @Qscores;
my @Sovscores;

my $thistag = $tag;

my $protlist = `cat $testfile`;
my @testlist = split (/\n/, $protlist);


################################################################
##	Generating Predictions
################################################################

# Probabilities are reformatted into usable file aligning the sequence
# with the corresponding predicted most likely secondary structure
my @dirarray = ($predir);
all_dnss_files(\@dirarray, $pssmdir, $dnssdir, \@testlist, $thistag);

timeprint ($logfile, "Evaluating predictions...");
my ($AvgQ3, $AvgSov) = score_dnss($dnssdir, \@testlist, $thistag);
next if check_err($AvgQ3, $logfile);


################################################################
################################################################

sub print_help {
	print "\nHelp Summary for sstest.pl script\n";
	print "Written by Matt Spencer\n";
	print "\nDescription:\n";
	print "This script trains and tests a DN for secondary structure prediction.\n";
	print "\nRequired input:\n";
	print "\t-pred	: The directory contains ss predictions (*.pred, *.prob).\n";
	print "\t-out	  : The directory where dnss formatted predictions are saved.\n";
	print "\t-list	: The proteins list to summarize.\n";
	print "\t-tag	: Identifying tag - the score file will be named as such.\n";
	print "\nOptions:\n";
	print "\t-help	: Print this help message.\n";
	print "\n\n";
}
