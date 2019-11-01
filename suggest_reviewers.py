"""
command: OMP_NUM_THREADS=12 python  nn.py --model avg --dim 300 --epochs 10 --ngrams 3 --share-vocab 1 --dropout 0.3 --gpu 0 --save-every-epoch 1 --nn-file s2/s2-filtered-aclweb.abs --load-file model_1.pt 
"""

import io
import sentencepiece as spm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from utils import Example, unk_string
from sacremoses import MosesTokenizer
import argparse
from models import Averaging, LSTM, load_model
import numpy as np
import sys
sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()

parser.add_argument("--reviewer-file", required=True, help="Reviewers that ")
parser.add_argument("--db-file", required=True, help="File (in s2 json format) of relevant papers from reviewers")
parser.add_argument("--reviewer-file", required=True, help="Reviewers that ")
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
# parser.add_argument("--load-file", help="filename to load a pretrained model.")
# parser.add_argument("--save-every-epoch", default=0, type=int, help="whether to save a checkpoint every epoch")
# parser.add_argument("--outfile", default="model", help="output file name")
# parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
# parser.add_argument("--delta", default=0.4, type=float, help="margin")
# parser.add_argument("--ngrams", default=0, type=int, help="whether to use character n-grams")
# parser.add_argument("--share-encoder", default=1, type=int, help="whether to share the encoder (LSTM only)")
# parser.add_argument("--share-vocab", default=1, type=int, help="whether to share the embeddings")
# parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling")
# parser.add_argument("--sp-model", help="SP model to load for evaluation")
# parser.add_argument("--nn-file", help="data file, one example per line")

args = parser.parse_args()

model, epoch = load_model(None, args)
model.eval()
assert not model.training
entok = MosesTokenizer(lang='en')

f = open(args.nn_file, "r")
lines = f.readlines()

data = []
for i in lines:
    p1 = " ".join(entok.tokenize(i.strip(), escape=False)).lower()
    if model.sp is not None:
        p1 = model.sp.EncodeAsPieces(p1)
        p1 = " ".join(p1)
    wp1 = Example(p1)
    wp1.populate_embeddings(model.vocab, model.zero_unk, args.ngrams)
    if len(wp1.embeddings) == 0:
        wp1.embeddings.append(model.vocab[unk_string])
    data.append(wp1)

embeddings = []
curr_batch = []
data = data[:5000] #for testing
for i in range(len(data)):
    curr_batch.append(data[i])
    if len(curr_batch) == 128:
        wx1, wl1 = model.torchify_batch(curr_batch)
        vecs = model.encode(wx1, wl1)
        vecs = vecs.detach().cpu().numpy()
        vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None] #normalize for NN search
        embeddings.extend([vecs[i,:] for i in range(vecs.shape[0])])
        curr_batch = []

if len(curr_batch) > 0:
    wx1, wl1 = model.torchify_batch(curr_batch)
    vecs = model.encode(wx1, wl1)
    vecs = vecs.detach().cpu().numpy()
    vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None] #normalize for NN search
    embeddings.extend([vecs[i,:] for i in range(vecs.shape[0])])

from scipy import spatial
tree = spatial.KDTree(embeddings)

#sample nns
for i in range(5):
    ex = data[i]
    emb = embeddings[i]
    dist, idx = tree.query(emb, k=4)
    idx = idx[1:]
    print("source: " + ex.sentence)
    for j in range(len(idx)):
        print("neighbor {0}, distance: {1}, text:".format(j+1, dist[j+1]) + data[idx[j]].sentence)
    print()
