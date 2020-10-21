#!/usr/bin/env perl
#
# grep for reviewers listed in a file
#
#  -i username-file ...... file of user names
#  -t .................... overwrite track info
#

use strict;
use Getopt::Std;

our ($opt_i,$opt_t);
getopts('i:t');

my $idfile = $opt_i || die "need to specify a file with user names!\n";

my %usernames = ();
open F,"<$idfile" || "cannot read from $idfile\n";
while (<F>){
    chomp;
    my ($user,$track) = split(/\t/);
    $usernames{$user} = $track;
}
close F;

while (<>){
    if (/\"startUsername\"\s*\:\s*"(.*?)\"/){
	my $user = $1;
	if (exists $usernames{$user}){
	    s/(\"areaChair\"\:)\s*false/$1 true/;
	    if ($opt_t){
		s/(\"track\"\:)\s*\".*?\"/$1 "$usernames{$user}"/;
	    }
	}
    }
    print;
}
