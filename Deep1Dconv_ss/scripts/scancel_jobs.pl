$start = $ARGV[0];
$end=$ARGV[1];

for($i=$start;$i<=$end;$i++)
{
	print "scancel $i\n";
	`scancel $i`;
}
