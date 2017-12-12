#!/usr/bin/perl -w
$num = @ARGV;
if($num != 1)
{
	die "Please provide the DeepProTa installation folder!\n";
	exit(-1);
}

$install_dir = $ARGV[0];

################Don't Change the code below##############

if (! -d $install_dir)
{
	die "can't find installation directory.\n";
}
if ( substr($install_dir, length($install_dir) - 1, 1) ne "/" )
{
	$install_dir .= "/"; 
}


if (prompt_yn("GANSS will be installed into <$install_dir> ")){

}else{
	die "The installation is cancelled!\n";
}
print "\n1. Start install GANSS into <$install_dir>\n\n";


@updatelist_py		=();
@updatelist_pl		=();
@updatelist_sh		=();
push @updatelist_py, 'Deep1Dconv_ss/scripts/train_deepcov_ss.py';
push @updatelist_py, 'Deep1Dconv_ss/scripts/train_deepcov_ss_all_train.py';
push @updatelist_py, 'lib/evaluate_deepcov_ss.py';

push @updatelist_pl, 'Deep1Dconv_ss/scripts/P1_generate_features.pl';
push @updatelist_pl, 'Deep1Dconv_ss_gan/scripts/P1_generate_features.pl';
push @updatelist_pl, 'lib/evaluation_dnss_prediction.pl';
push @updatelist_pl, 'lib/DN_SSpred.pm';

push @updatelist_sh, 'examples/test_deepss.sh';

$file_index=1;
#GLOBAL_PATH='/storage/htc/bdm/jh7x3/GANSS/';
### update python
foreach my $file (@updatelist_py) {
	$file2update=$install_dir.$file;
	
	$check_log ='GLOBAL_PATH=';
	open(IN,$file2update) || die "Failed to open file $file2update\n";
	open(OUT,">$file2update.tmp") || die "Failed to open file $file2update.tmp\n";
	while(<IN>)
	{
		$line = $_;
		chomp $line;

		if(index($line,$check_log)>=0)
		{
      $file_index++;
			print "$file_index. Setting ".$file2update."\n";
			print "\t--- Current ".$line."\n";
			print "\t--- Change to ".substr($line,0,index($line, '=')+1)." \'".$install_dir."\';\n\n\n";
			print OUT substr($line,0,index($line, '=')+1)."\'".$install_dir."\';\n";
		}else{
			print OUT $line."\n";
		}
	}
	close IN;
	close OUT;
	system("mv $file2update.tmp $file2update");
	system("chmod 755  $file2update");
}

#BEGIN { $GLOBAL_PATH = "/storage/htc/bdm/jh7x3/GANSS/"; }
### update perl
foreach my $file (@updatelist_pl) {
	$file2update=$install_dir.$file;
	
	$check_log ='BEGIN { $GLOBAL_PATH =';
	open(IN,$file2update) || die "Failed to open file $file2update\n";
	open(OUT,">$file2update.tmp") || die "Failed to open file $file2update.tmp\n";
	while(<IN>)
	{
		$line = $_;
		chomp $line;

		if(index($line,$check_log)>=0)
		{
      $file_index++;
			print "$file_index. Setting ".$file2update."\n";
			print "\t--- Current ".$line."\n";
			print "\t--- Change to ".substr($line,0,index($line, '=')+1)." \'".$install_dir."\';\n\n\n";
			print OUT substr($line,0,index($line, '=')+1)." \'".$install_dir."\'; }\n";
		}else{
			print OUT $line."\n";
		}
	}
	close IN;
	close OUT;
	system("mv $file2update.tmp $file2update");
	system("chmod 755  $file2update");
}


#GLOBAL_PATH=/storage/htc/bdm/jh7x3/GANSS
### update sh
foreach my $file (@updatelist_sh) {
	$file2update=$install_dir.$file;
	
	$check_log ='GLOBAL_PATH=';
	open(IN,$file2update) || die "Failed to open file $file2update\n";
	open(OUT,">$file2update.tmp") || die "Failed to open file $file2update.tmp\n";
	while(<IN>)
	{
		$line = $_;
		chomp $line;

		if(index($line,$check_log)>=0)
		{
      $file_index++;
			print "$file_index. Setting ".$file2update."\n";
			print "\t--- Current ".$line."\n";
			print "\t--- Change to ".substr($line,0,index($line, '=')+1)." \'".$install_dir."\';\n\n\n";
			print OUT substr($line,0,index($line, '=')+1).$install_dir."\n";
		}else{
			print OUT $line."\n";
		}
	}
	close IN;
	close OUT;
	system("mv $file2update.tmp $file2update");
	system("chmod 755  $file2update");
}

##### check if all data is available

@check_data		=();
push @check_data, 'DNSS_dataset/caspfasta';
push @check_data, 'DNSS_dataset/chains';
push @check_data, 'DNSS_dataset/dssp';
push @check_data, 'DNSS_dataset/fasta';
push @check_data, 'DNSS_dataset/lists';
push @check_data, 'DNSS_dataset/pssm';
push @check_data, 'DNSS_dataset/ssa';
push @check_data, 'GANSS_Datasets/features_win15_no_atch_aa';

foreach my $file (@check_data) {
	$file2check=$install_dir.$file;
  if(!(-d $file2check))
  {
    die "Failed to find $file2check, please check the installation file or contact Jie Hou<jh7x3@mail.missouri.edu>\n";
  }
}
@check_files		=();
push @check_files, 'GANSS_Datasets/adj_dncon-test.lst';
push @check_files, 'GANSS_Datasets/dncov_training.list';
push @check_files, 'GANSS_Datasets/dncov_validation.list';
    
foreach my $file (@check_files) {
	$file2check=$install_dir.$file;
  if(!(-e $file2check))
  {
    die "Failed to find $file2check, please check the installation file or contact Jie Hou<jh7x3@mail.missouri.edu>\n";
  }
} 


$file_index++;
print "\n$file_index. Installation finished!\n";





sub prompt_yn {
  my ($query) = @_;
  my $answer = prompt("$query (Y/N): ");
  return lc($answer) eq 'y';
}
sub prompt {
  my ($query) = @_; # take a prompt string as argument
  local $| = 1; # activate autoflush to immediately show the prompt
  print $query;
  chomp(my $answer = <STDIN>);
  return $answer;
}
