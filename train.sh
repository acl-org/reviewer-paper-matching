#!/bin/sh
source activate acl-review

bash download_sts17.sh
python tokenize_abstracts.py --infile res/acl-anthology.json --outfile res/abstracts.txt
python sentencepiece_abstracts.py --infile res/abstracts.txt --vocab-size 20000 \
                                  --model-name res/abstracts.sp.20k --outfile res/abstracts.20k.sp.txt 
python -u train_similarity.py --data-file res/abstracts.20k.sp.txt \
                              --model avg --dim 1024 --epochs 20 --ngrams 0 --share-vocab 1 --dropout 0.3 \
                              --outfile res/similarity-model.pt --batchsize 64 --megabatch-size 1 \
                              --megabatch-anneal 10 --seg-length 1 \
                              --sp-model res/abstracts.sp.20k.model 2>&1 | tee res/training.log

