#!/usr/bin/perl
# Script: 	sstest.pl
# Author: 	Matt Spencer
# Made:		9/25/13
# Last Mod:	11/7/13
#
#
# Input:
#
# Dependencies:

use lib "../lib";
use strict;
use Time qw(formatted_localtime);
use DN_SSpred qw(newDN timeprint sort_scores score_dnss check_err); 
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
$iters = 1 unless ($iters);
$pssm = 1 unless (defined($pssm));
$atch = 1 unless (defined($atch));
$seq = 0 unless (defined($seq));
$wind = 7 unless ($wind);
$boost = 1 unless ($boost);
$skipx = 0 unless ($skipx);
$bound = 1 unless (defined($bound));
$reduced = 0 unless ($reduced);
my $target = 3*$boost;
$arch = "200,200,$target" unless ($arch);
my @archinfo = split(/,/, $arch);
#my $lwind = 17;
#my $larch = "600,600,600,250,3";
#####################################

my $trainfile = "../data/lists/dncon-train.lst";
my $testfile = "../data/lists/dncon-test.lst";
my $ssadir = "../data/ssa/";
my $pssmdir = "../data/pssm/";
my $overdir = "/home/mattcspencer/DNSA/data/ss_pred/";
$outdir = $overdir . $outdir;
my $logfile = $outdir . "log.txt";

my @features = ($pssm, $atch, $seq, $bound, $skipx);
my @options = (\@features, $wind, $boost, $reduced);
my @dirs = ($ssadir, $pssmdir);

`mkdir $overdir` unless (-d $overdir);
`mkdir $outdir` unless (-d $outdir);

timeprint($logfile, "Performing SS train & test using the parameters:\nTag: $tag\nOutdir: $outdir\nIterations: $iters\nReduced: $reduced\nReplace: $replace\nSkip X: $skipx\n\nWindow: $wind\nPssm: $pssm\nAtch: $atch\nSeq: $seq\nArch: $arch\nBoost: $boost\nBound: $bound\n");

`rm -r $outdir/*` if ($replace);

my @Qscores;
my @Sovscores;
for (my $ii=0; $ii<$iters; $ii++){
	my $iterdir = $outdir . "iter$ii/";
	`mkdir $iterdir` unless (-d $iterdir);
	my $thistag = $tag . "_$ii";

	my $protlist = `cat $trainfile`;
	my @trainlist = split (/\n/, $protlist);
	my $protlist = `cat $testfile`;
	my @testlist = split (/\n/, $protlist);

	my ($predir, $dnssdir) = 
		newDN(	$logfile, $iterdir, \@trainlist, \@testlist, 
			\@dirs, \@options, \@archinfo, "iter$ii");

	my @predirs = ($predir);

	timeprint ($logfile, "Evaluating predictions...");

	my ($AvgQ3, $AvgSov) = score_dnss($dnssdir, \@testlist, $thistag);
	next if check_err($AvgQ3, $logfile);

	push (@Qscores, $AvgQ3);
	push (@Sovscores, $AvgSov);
}

my ($index, @scoreorder) = sort_scores(\@Qscores, \@Sovscores);

my $bestdir = $outdir . "iter$index/";
timeprint($logfile, "Best trial: $bestdir");
timeprint($logfile, "Scores obtained: ");
foreach (@scoreorder){
	my ($iter, $Q3, $Sov) = @{ $_ };
	timeprint($logfile, "Trial: $iter\tQ3: $Q3\tSov: $Sov");
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
	print "\t-bound	: Include boundary features.\n";
	print "\t-arch	: Indicate the architecture to use.\n";
	print "\t-iter	: Indicate the number of iterations to attempt.\n";
	print "\t-help	: Print this help message.\n";
	print "\n\n";
}
