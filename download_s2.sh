#!/usr/bin/bash

manifest_file=$1
split_num=$2

# Work out lines per file
total_lines=$(wc -l <${manifest_file})
((lines_per_file = (total_lines + split_num - 1) / split_num))

# Split the actual file, maintaining lines
mkdir manifests
split --lines=${lines_per_file} ${manifest_file} manifests/manifest.

# Debug information
echo "Total files = ${total_lines}"
echo "Files per manifest = ${lines_per_file}"    
wc -l manifests/manifest.*

mkdir s2
mkdir scratch
touch scratch/acl-anthology.json
chunk=1

for FILE in manifests/*
do
    echo "Downloading and filtering Semantic Scholar dump part ${chunk} of ${split_num}"
    wget -q -P s2 -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-12-01/ -i ${FILE}
    zcat s2/s2-corpus*.gz | grep aclweb.org >> scratch/acl-anthology.json
    rm -f s2/*
    chunk=$(( $chunk + 1 ))
done

rm -rf s2
rm -rf manifests
