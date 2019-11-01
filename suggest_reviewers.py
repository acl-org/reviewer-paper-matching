"""
command: OMP_NUM_THREADS=12 python  nn.py --model avg --dim 300 --epochs 10 --ngrams 3 --share-vocab 1 --dropout 0.3 --gpu 0 --save-every-epoch 1 --nn-file s2/s2-filtered-aclweb.abs --load-file model_1.pt 
"""

import io
import sentencepiece as spm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import json
from utils import Example, unk_string
from sacremoses import MosesTokenizer
import argparse
from models import Averaging, LSTM, load_model
import numpy as np
import sys
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()

parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line file of submissions")
parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
parser.add_argument("--reviewer_file", type=str, required=True, help="A file of reviewer files or IDs that can review this time")
parser.add_argument("--filter_field", type=str, default="Name", help="Which field to filter on")
parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
parser.add_argument("--save_matrix", help="A filename for where to save the similarity matrix")
parser.add_argument("--load_matrix", help="A filename for where to load the cached similarity matrix")
parser.add_argument("--ngrams", default=0, type=int, help="whether to use character n-grams")

# parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
# parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
# parser.add_argument("--model", default="avg", choices=["avg", "lstm"], help="type of base model to train.")
# parser.add_argument("--grad-clip", default=5., type=float, help='clip threshold of gradients')
# parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
# parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
# parser.add_argument("--dropout", default=0., type=float, help="dropout rate")
# parser.add_argument("--batchsize", default=128, type=int, help="size of batches")
# parser.add_argument("--megabatch-size", default=60, type=int, help="number of batches in megabatch")
# parser.add_argument("--megabatch-anneal", default=150., type=int, help="rate of megabatch annealing in terms of "
#                                                                        "number of batches to process before incrementing")
# parser.add_argument("--pool", default="mean", choices=["mean", "max"], help="type of pooling")
# parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")

# parser.add_argument("--save-every-epoch", default=0, type=int, help="whether to save a checkpoint every epoch")
# parser.add_argument("--outfile", default="model", help="output file name")
# parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
# parser.add_argument("--delta", default=0.4, type=float, help="margin")
# parser.add_argument("--share-encoder", default=1, type=int, help="whether to share the encoder (LSTM only)")
# parser.add_argument("--share-vocab", default=1, type=int, help="whether to share the embeddings")
# parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling")
# parser.add_argument("--sp-model", help="SP model to load for evaluation")
# parser.add_argument("--nn-file", help="data file, one example per line")

args = parser.parse_args()

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')

def print_progress(i):
    if i != 0 and i % BATCH_SIZE == 0:
        sys.stderr.write('.')
        if int(i/BATCH_SIZE) % 50 == 0:
            print(i, file=sys.stderr)

def create_embeddings(model, examps):
    """Embed textual examples

    :param examps: A list of text to embed
    :return: A len(examps) by embedding size numpy matrix of embeddings
    """
    # Preprocess examples
    print(f'Preprocessing {len(examps)} examples (.={BATCH_SIZE} examples)', file=sys.stderr)
    data = []
    for i, line in enumerate(examps):
        p1 = " ".join(entok.tokenize(line, escape=False)).lower()
        if model.sp is not None:
            p1 = model.sp.EncodeAsPieces(p1)
            p1 = " ".join(p1)
        wp1 = Example(p1)
        wp1.populate_embeddings(model.vocab, model.zero_unk, args.ngrams)
        if len(wp1.embeddings) == 0:
            wp1.embeddings.append(model.vocab[unk_string])
        data.append(wp1)
        print_progress(i)
    print("", file=sys.stderr)
    # Create embeddings
    print(f'Embedding {len(examps)} examples (.={BATCH_SIZE} examples)', file=sys.stderr)
    embeddings = np.zeros( (len(examps), model.args.dim) )
    for i in range(0, len(data), BATCH_SIZE):
        max_idx = min(i+BATCH_SIZE,len(data))
        curr_batch = data[i:max_idx]
        wx1, wl1 = model.torchify_batch(curr_batch)
        vecs = model.encode(wx1, wl1)
        vecs = vecs.detach().cpu().numpy()
        vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None] #normalize for NN search
        embeddings[i:max_idx] = vecs
        print_progress(i)
    print("", file=sys.stderr)
    return embeddings

def calc_similarity_matrix(model, db, quer):
    db_emb = create_embeddings(model, db)
    quer_emb = create_embeddings(model, quer)
    print(f'Performing similarity calculation', file=sys.stderr)
    return np.matmul(quer_emb, np.transpose(db_emb))

with open(args.submission_file, "r") as f:
    submission_abs = [x.strip() for x in f]
with open(args.db_file, "r") as f:
    db = [json.loads(x) for x in f] # for debug
    db_abs = [x['paperAbstract'] for x in db]

model, epoch = load_model(None, args.model_file)
model.eval()
assert not model.training

if args.load_matrix:
    mat = np.load(args.load_matrix)
else:
    mat = calc_similarity_matrix(model, db_abs, submission_abs)
    if args.save_matrix:
        np.save(args.save_matrix, mat)

for i, query in enumerate(submission_abs):
    scores = mat[i]
    best_idxs = scores.argsort()[-5:][::-1]
    print('-----------------------------------------------------')
    print('*** Paper Abstract')
    print(query)
    print('\n *** Similar Paper Abstracts')
    for idx in best_idxs:
        print(f'# Score {scores[idx]}\n{db[idx]}')
    print()
