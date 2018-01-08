#!/usr/bin/perl -w

$numArgs = @ARGV;
if($numArgs != 3)
{   
	print "the number of parameters is not correct!\n";
	exit(1);
}

$wordir		= "$ARGV[0]";
$start	= "$ARGV[1]";
$end	= "$ARGV[2]";

chdir($wordir);

for($i=$start;$i<=$end;$i++)
{
  $batchfile = "P1_run_sbatch_$i.sh";
  if(-e $batchfile)
  {
    $found = 0;
    if($i % 5==0)
    {
       print "let's wait......\n";	
       sleep(5);
    }
    if($found == 0)
    {
      print "Running $batchfile\n";
      `sbatch $batchfile`;
      #`sh $batchfile`;
      
    }
  }
}

  
  
  
