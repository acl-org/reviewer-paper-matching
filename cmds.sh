python -u train_similarity.py --data-file scratch/acl-anthology-training.tsv --model avg --dim 300 --epochs 20 --ngrams 3 --share-vocab 1 --dropout 0.3 --outfile sim-3-300
python -u train_similarity.py --data-file scratch/acl-anthology-training.tsv --model avg --dim 1024 --epochs 20 --ngrams 3 --share-vocab 1 --dropout 0.3 --outfile sim-3-1024
python -u train_similarity.py --data-file scratch/acl-anthology-training.tsv --model avg --dim 300 --epochs 20 --ngrams 4 --share-vocab 1 --dropout 0.3 --outfile sim-4-300
python -u train_similarity.py --data-file scratch/acl-anthology-training.tsv --model avg --dim 1024 --epochs 20 --ngrams 4 --share-vocab 1 --dropout 0.3 --outfile sim-4-1024
