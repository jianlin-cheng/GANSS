#!/usr/bin/perl
# Script: 	satest.pl
# Author: 	Matt Spencer
# Made:		9/25/13
# Last Mod:	11/5/14
#
# This script is used to train a DN using a particular 
# configuration of parameters, and then to test the resulting
# DN on the test proteins to see how effective it is.
#
# Input: 
#	Tag		Name of trial, dictates name of overall score file
#	Outdir		Directory where ouput files will be stored
#
# Optional Input: (defaults specified below)
#	-r	Replace existing files in specified outdir
#	-x	Deal with X residues differently (see -help)
#		(this is not known to be helpful, generally leave as default)
#	-wind	Indicate the window size to use
#	-boost	Indicate the size of boost window (not implemented - use default)
#	-pssm	Include pssm features (almost a must)
#	-atch	Include atchley factors
#	-seq	Include sequence hard-coded features
#	-bound	Include Boundary features
#	-arch	Indicate the DN architecture to use
#	-iter	Indicate the number of attempts
#	-help	Print Help message
#

use strict;
use lib '/home/mattcspencer/DNSA/lib';
use Time qw(formatted_localtime);
use DN_SApred qw(newDN timeprint sort_scores score_dnsa check_err); 
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

if ($help || !$tag || !$outdir){
	print_help();
	exit;
}

$outdir .= "/" unless ($outdir =~ m/\/$/);

#####################################
#########     DEFAULTS     ##########
$iters = 1 unless ($iters);
$pssm = 1 unless (defined($pssm));
$atch = 1 unless (defined($atch));
$seq = 0 unless (defined($seq));
$wind = 13 unless ($wind);
$boost = 1 unless ($boost);
$skipx = 0 unless ($skipx);
$bound = 1 unless (defined($bound));
$reduced = 0 unless ($reduced);
my $target = 1*$boost;
$arch = "200,200,$target" unless ($arch);
my @archinfo = split(/,/, $arch);
#####################################

my $trainfile = "../data/lists/dncon-train.lst";  #trains useing these prots
my $testfile = "../data/lists/dncon-test.lst";    #tests using these prots
my $ssadir = "../data/ssa/";
my $pssmdir = "../data/pssm/";
my $overdir = "/home/mattcspencer/DNSA/data/sa_pred/";
$outdir = $overdir . $outdir;
my $logfile = $outdir . "log.txt";

my @features = ($pssm, $atch, $seq, $bound, $skipx);
my @options = (\@features, $wind, $boost, $reduced);
my @dirs = ($ssadir, $pssmdir);

`mkdir $overdir` unless (-d $overdir);
`mkdir $outdir` unless (-d $outdir);

timeprint($logfile, "Performing SS train & test using the parameters:\nTag: $tag\nOutdir: $outdir\nIterations: $iters\nReduced: $reduced\nReplace: $replace\nSkip X: $skipx\n\nWindow: $wind\nPssm: $pssm\nAtch: $atch\nSeq: $seq\nArch: $arch\nBoost: $boost\nBound: $bound\n");

`rm -r $outdir/*` if ($replace);

my @RMSDscores;
my @Pearscores;
for (my $ii=0; $ii<$iters; $ii++){
	my $iterdir = $outdir . "iter$ii/";
	`mkdir $iterdir` unless (-d $iterdir);
	my $thistag = $tag . "_$ii";

	# Load lists of proteins
	my $protlist = `cat $trainfile`;
	my @trainlist = split (/\n/, $protlist);
	my $protlist = `cat $testfile`;
	my @testlist = split (/\n/, $protlist);

	my $dnsadir;

	# The success file is made to indicate that the iteration was 
	# successful. This way multiple attempts at running the same
	# program twice will allow it to save time by skipping 
	# processes that were already done. The success file is not
	# necessary for normal functionality and can be removed.
	my $successfile = $iterdir . "success.txt";

if (! (-f $successfile)) { 

	($dnsadir) = 
		newDN(	$logfile, $iterdir, \@trainlist, \@testlist, 
			\@dirs, \@options, \@archinfo, "iter$ii");
	`echo "Success!\n" > $successfile`;
}
else {
	timeprint ($logfile, "newDN previously accomplished.\n");
	$dnsadir = $iterdir . "dnsa/";
}	


	timeprint ($logfile, "Evaluating predictions...");

	my ($AvgRMSD, $AvgPear, $AvgQ10, $AvgQperc) = score_dnsa($dnsadir, \@testlist, $thistag);
	next if check_err($AvgRMSD, $logfile);

	push (@RMSDscores, $AvgRMSD);
	push (@Pearscores, $AvgPear);
	timeprint($logfile, "RMSD: $AvgRMSD\tPear: $AvgPear\tQ10: $AvgQ10\tQ%: $AvgQperc");
}

my ($index, @scoreorder) = sort_scores(\@RMSDscores, \@Pearscores);

my $bestdir = $outdir . "iter$index/";
timeprint($logfile, "Best trial: $bestdir");
timeprint($logfile, "Scores obtained: ");
foreach (@scoreorder){
	my ($iter, $RMSD, $Pear) = @{ $_ };
	timeprint($logfile, "Trial: $iter\tRMSD: $RMSD\tPear: $Pear");
}
print "\n\n";


################################################################
################################################################

sub print_help {
	print "\nHelp Summary for sstest.pl script\n";
	print "Written by Matt Spencer\n";
	print "\nDescription:\n";
	print "This script trains and tests a DN for secondary structure prediction.\n";
	print "\nRequired input:\n";
	print "\t-tag	: Identifying tag - the score file will be named as such.\n";
	print "\t-out	: Subdirectory of data/ss_pred/ to save intermediate files to.\n";
	print "\nOptions:\n";
	print "\t-r	: Indicates that previously existing files will be replaced.\n";
	print "\t-x	: Deal with X res (0: keep, 1: skip lines, 2: skip windows)\n";
	print "\t-red	: Only predict non-gap residues.\n";
	print "\t-wind	: Indicate the window size to use.\n";
	print "\t-boost	: Indicate the size of boost window.\n";
	print "\t-pssm	: Include pssm features.\n";
	print "\t-atch	: Include atchley factors.\n";
	print "\t-seq	: Include seq features.\n";
	print "\t-bound	: Inclu1de boundary features.\n";
	print "\t-arch	: Indicate the architecture to use.\n";
	print "\t-iter	: Indicate the number of iterations to attempt.\n";
	print "\t-help	: Print this help message.\n";
	print "\n\n";
}
