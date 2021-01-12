#!/bin/sh
source activate acl-review

bash download_sts17.sh
python reviewer-paper-matching/tokenize_abstracts.py --infile scratch/acl-anthology.json --outfile scratch/abstracts.txt
python reviewer-paper-matching/sentencepiece_abstracts.py --infile scratch/abstracts.txt --vocab-size 20000 \
                                  --model-name scratch/abstracts.sp.20k --outfile scratch/abstracts.20k.sp.txt 
python -u reviewer-paper-matching/train_similarity.py --data-file scratch/abstracts.20k.sp.txt \
                              --model avg --dim 1024 --epochs 20 --ngrams 0 --share-vocab 1 --dropout 0.3 \
                              --outfile scratch/similarity-model.pt --batchsize 64 --megabatch-size 1 \
                              --megabatch-anneal 10 --seg-length 1 \
                              --sp-model scratch/abstracts.sp.20k.model 2>&1 | tee logs/training.log

