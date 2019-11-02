"""
command: OMP_NUM_THREADS=12 python  nn.py --model avg --dim 300 --epochs 10 --ngrams 3 --share-vocab 1 --dropout 0.3 --gpu 0 --save-every-epoch 1 --nn-file s2/s2-filtered-aclweb.abs --load-file model_1.pt 
"""

import json
from utils import Example, unk_string
from sacremoses import MosesTokenizer
import argparse
from models import load_model
from collections import defaultdict
import numpy as np
import sys
sys.setrecursionlimit(10000)

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


def calc_reviewer_db_mapping(reviewers, db, filter_field):
    print(f'Calculating reviewer-paper mapping for {len(reviewers)} reviewers and {len(db)} papers', file=sys.stderr)
    mapping = np.zeros( (len(reviewers), len(db)) )
    reviewer_id_map = defaultdict(lambda: [])
    for i, reviewer in enumerate(reviewers):
        reviewer_id_map[reviewer].append(i)
    for j, entry in enumerate(db):
        for cols in entry['authors']:
            if cols[filter_field] in reviewer_id_map:
                for i in reviewer_id_map[cols[filter_field]]:
                    mapping[i,j] = 1
    return mapping


def calc_aggregate_reviewer_score(rdb, scores, operator='max'):
    """Calculate the aggregate reviewer score for one paper

    :param rdb: Reviewer DB. Numpy matrix of reviewers x DB_size
    :param simscores: Similarity scores between the current paper and the DB. Numpy array of length DB_size
    :param operator: Which operator to apply (max, weighted_topK)
    :return: Numpy matrix of length reviewers indicating the score for that reviewer
    """
    INVALID_SCORE = 0
    scored_rdb = rdb * scores.reshape((1, scores.shape[0])) + (1-rdb) * INVALID_SCORE
    if operator == 'max':
        agg = np.amax(scored_rdb, axis=1)
    elif operator.startswith('weighted_top'):
        k = int(operator[12:])
        weighting = np.reshape(1/np.array(range(1, k+1)), (1,k))
        scored_rdb.sort(axis=1)
        topk = scored_rdb[:,-k:]
        # print(topk)
        agg = (topk*weighting).sum(axis=1)
    else:
        raise ValueError(f'Unknown operator {operator}')
    return agg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line file of submissions")
    parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A file of reviewer files or IDs that can review this time")
    parser.add_argument("--filter_field", type=str, default="Name", help="Which field to filter on")
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--save_paper_matrix", help="A filename for where to save the paper similarity matrix")
    parser.add_argument("--load_paper_matrix", help="A filename for where to load the cached paper similarity matrix")
    parser.add_argument("--ngrams", default=0, type=int, help="whether to use character n-grams")

    args = parser.parse_args()

    # Load the data
    with open(args.submission_file, "r") as f:
        submission_abs = [x.strip() for x in f]
        submissions = [{'paperAbstract': x} for x in submission_abs]
    with open(args.reviewer_file, "r") as f:
        reviewer_names = [x.strip() for x in f]
    with open(args.db_file, "r") as f:
        db = [json.loads(x) for x in f]  # for debug
        db_abs = [x['paperAbstract'] for x in db]
    rdb = calc_reviewer_db_mapping(reviewer_names, db, 'name')

    # Calculate or load paper similarity matrix
    if args.load_paper_matrix:
        mat = np.load(args.load_paper_matrix)
        assert(mat.shape[0] == len(submission_abs) and mat.shape[1] == len(db_abs))
    else:
        print('Loading model', file=sys.stderr)
        model, epoch = load_model(None, args.model_file)
        model.eval()
        assert not model.training
        mat = calc_similarity_matrix(model, db_abs, submission_abs)
        if args.save_paper_matrix:
            np.save(args.save_paper_matrix, mat)

    reviewer_scores = np.zeros( (len(submissions), len(reviewer_names)) )
    for i, query in enumerate(submissions):
        scores = mat[i]
        best_idxs = scores.argsort()[-5:][::-1]
        print('-----------------------------------------------------')
        print('*** Paper Abstract')
        print(query)
        print('\n*** Similar Paper Abstracts')
        for idx in best_idxs:
            print(f'# Score {scores[idx]}\n{db_abs[idx]}')
        print()

        reviewer_scores[i] = calc_aggregate_reviewer_score(rdb, scores, 'weighted_top3')
        best_reviewers = reviewer_scores[i].argsort()[-5:][::-1]
        print('*** Best Matched Reviewers')
        for idx in best_reviewers:
            print(f'# {reviewer_names[idx]} (Score {reviewer_scores[i][idx]})')
        print()

