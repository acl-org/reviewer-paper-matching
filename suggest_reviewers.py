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
import cvxpy as cp
import sys
sys.setrecursionlimit(10000)

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')


def print_progress(i, mod_size=BATCH_SIZE):
    if i != 0 and i % mod_size == 0:
        sys.stderr.write('.')
        if int(i/mod_size) % 50 == 0:
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
        wp1.populate_embeddings(model.vocab, model.zero_unk, model.args.ngrams)
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


def calc_aggregate_reviewer_score(rdb, all_scores, operator='max'):
    """Calculate the aggregate reviewer score for one paper

    :param rdb: Reviewer DB. NP matrix of reviewers by DB papers
    :param scores: NP matrix of similarity scores between the current papers (rows) and the DB papers (columns)
    :param operator: Which operator to apply (max, weighted_topK)
    :return: Numpy matrix of length reviewers indicating the score for that reviewer
    """
    agg = np.zeros( (all_scores.shape[0], len(rdb)) )
    print(f'Calculating aggregate scores for {all_scores.shape[0]} examples (.=10 examples)', file=sys.stderr)
    for i in range(all_scores.shape[0]):
        scores = all_scores[i]
        INVALID_SCORE = 0
        scored_rdb = rdb * scores.reshape((1, scores.shape[0])) + (1-rdb) * INVALID_SCORE
        if operator == 'max':
            agg[i] = np.amax(scored_rdb, axis=1)
        elif operator.startswith('weighted_top'):
            k = int(operator[12:])
            weighting = np.reshape(1/np.array(range(1, k+1)), (1,k))
            scored_rdb.sort(axis=1)
            topk = scored_rdb[:,-k:]
            # print(topk)
            agg[i] = (topk*weighting).sum(axis=1)
        else:
            raise ValueError(f'Unknown operator {operator}')
        print_progress(i, mod_size=10)
    return agg


def create_suggested_assignment(reviewer_scores, reviews_per_paper=3, max_papers_per_reviewer=5):
    num_pap, num_rev = reviewer_scores.shape
    if num_rev*max_papers_per_reviewer < num_pap*reviews_per_paper:
        raise ValueError(f'There are not enough reviewers ({num_rev}) review all the papers ({num_pap})'
                         f' given a constraint of {reviews_per_paper} reviews per paper and'
                         f' {max_papers_per_reviewer} reviews per reviewer')
    assignment = cp.Variable(shape=reviewer_scores.shape, boolean=True)
    rev_constraint = cp.sum(assignment, axis=0) <= max_papers_per_reviewer
    pap_constraint = cp.sum(assignment, axis=1) == reviews_per_paper
    total_sim = cp.sum(cp.multiply(reviewer_scores, assignment))
    constraints = [rev_constraint, pap_constraint]
    assign_prob = cp.Problem(cp.Maximize(total_sim), constraints)
    assign_prob.solve(solver=cp.GLPK_MI)
    return assignment.value, assign_prob.value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line file of submissions")
    parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A file of reviewer files or IDs that can review this time")
    parser.add_argument("--filter_field", type=str, default="Name", help="Which field to filter on")
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--save_paper_matrix", help="A filename for where to save the paper similarity matrix")
    parser.add_argument("--load_paper_matrix", help="A filename for where to load the cached paper similarity matrix")
    parser.add_argument("--max_papers_per_reviewer", default=5, type=int, help="How many papers, maximum, to assign to each reviewer")
    parser.add_argument("--reviews_per_paper", default=3, type=int, help="How many reviews to assign to each paper")
    parser.add_argument("--output_type", default="json", type=str, help="What format of output to produce (json/text)")

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

    # Calculate reviewer scores based on paper similarity scores
    reviewer_scores = np.zeros( (len(submissions), len(reviewer_names)) )
    print('Calculating aggregate reviewer scores', file=sys.stderr)
    reviewer_scores = calc_aggregate_reviewer_score(rdb, mat, 'weighted_top3')

    # Calculate a reviewer assignment based on the constraints
    print('Calculating assignment of reviewers', file=sys.stderr)
    assignment, assignment_score = create_suggested_assignment(reviewer_scores,
                                             max_papers_per_reviewer=args.max_papers_per_reviewer,
                                             reviews_per_paper=args.reviews_per_paper)

    # Print out the results
    for i, query in enumerate(submission_abs):
        scores = mat[i]
        best_idxs = scores.argsort()[-5:][::-1]
        best_reviewers = reviewer_scores[i].argsort()[-5:][::-1]
        assigned_reviewers = assignment[i].argsort()[-args.reviews_per_paper:][::-1]

        if args.output_type == 'json':
            ret_dict = {'paperAbstract': query}
            ret_dict['similarAbstracts'] = [(scores[idx], db_abs[idx]) for idx in best_idxs]
            ret_dict['topSimReviewers'] = [(reviewer_scores[i][idx], reviewer_names[idx]) for idx in best_reviewers]
            ret_dict['assignedReviewers'] = [(reviewer_scores[i][idx], reviewer_names[idx]) for idx in best_reviewers]
            print(json.dumps(ret_dict))
        elif args.output_type == 'text':
            print('----------------------------------------------')
            print('*** Paper Abstract')
            print(query)
            print('\n*** Similar Paper Abstracts')
            for idx in best_idxs:
                print(f'# Score {scores[idx]}\n{db_abs[idx]}')
            print()
            print('\n*** Best Matched Reviewers')
            for idx in best_reviewers:
                print(f'# {reviewer_names[idx]} (Score {reviewer_scores[i][idx]})')
            print('\n*** Assigned Reviewers')
            for idx in assigned_reviewers:
                print(f'# {reviewer_names[idx]} (Score {reviewer_scores[i][idx]})')
            print()
        else:
            raise ValueError(f'Illegal output_type {args.output_type}')



