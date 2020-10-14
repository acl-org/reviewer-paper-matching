#!/usr/bin/env perl
#
# grep for papers from the Semantic Scholar DB
#
# -i idfile ......... get papers with IDs listed in <idfile>
# -q regex .......... get paper records that match a regex query
#

use Getopt::Std;

our %opts = ();

getopt('q:i:', \%opts);

my $idfile = $opts{i};
my $query  = $opts{q};

my %paperids = ();
open F,"<$idfile" || "cannot read from $idfile\n";
while (<F>){
    chomp;
    $paperids{$_}++;
}
close F;

print STDERR "read ",scalar keys %paperids, " ids\n";

my %foundids = ();
while (<>){
    if ($idfile){
	if (/\"id\"\:\"(.*?)\"/){
	    if (exists $paperids{$1}){
		$foundids{$1}++;
		print;
		next;
	    }
	}
    } 
    if ($query){
	print if (/$query/);
    }
}

foreach (keys %paperids){
    unless ($foundids{$_}){
	print STDERR "WARNING: paper not found $_\n";
    }
}
