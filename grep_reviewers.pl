#!/usr/bin/env perl
#
# grep for reviewers listed in a file
#
#  -i username-file ...... file of user names
#  -v .................... reverse (print all names that are not listed)
#

use strict;
use Getopt::Std;

our ($opt_i, $opt_v);
getopts('i:v');

my $idfile = $opt_i || die "need to specify a file with user names!\n";

my %usernames = ();
open F,"<$idfile" || "cannot read from $idfile\n";
while (<F>){
    chomp;
    $usernames{$_}++;
}
close F;

while (<>){
    if (/\"startUsername\"\s*\:\s*"(.*?)\"/){
	if (exists $usernames{$1}){
	    print unless ($opt_v);
	}
	elsif ($opt_v){
	    print;
	}
    }
}
