#!/usr/bin/env perl
#
# add the COI track for some reviewers
# (Multidisciplinary and AC COI)
#
#  -i username-file ...... file of user names
#

use strict;
use Getopt::Std;

our ($opt_i);
getopts('i:');

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
	    s/(\"track\"\:\s*\")/${1}Multidisciplinary and AC COI:/;
	}
    }
    print;
}