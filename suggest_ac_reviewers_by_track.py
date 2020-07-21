"""
This is a variant of the suggest_reviewers script tailored for assignment of papers to area chairs.

It was run as follows for EMNLP 2020:

    python3 suggest_ac_reviewers_by_track.py \
        --submission_file=$dir/submissions.jsonl \
        --db_file=scratch/acl-anthology.json \
        --reviewer_file=$dir/reviewers.jsonl \
        --model_file=scratch/similarity-model.pt \
        --max_papers_per_reviewer=20 \
        --reviews_per_paper=1 \
        --suggestion_file=$dir/ac-assignments.jsonl \
        --bid_file=$dir/cois.npy \
        --paper_matrix=$dir/ac-paper_matrix.npy \
        --aggregate_matrix=$dir/ac-agg_matrix.npy  \
        --quota_file $dir/ac_quotas.csv \
        --min_papers_per_reviewer=10 

The suggestions file can be converted to SOFTCONF format using

    python3 softconf_package.py \
         --suggestion_file $dir/ac-assignments.jsonl \
         --softconf_file $dir/ac-start-assignments.csv \
         --split_by_track
"""

import json
from model_utils import Example, unk_string
from sacremoses import MosesTokenizer
import argparse
from models import load_model
import numpy as np
import cvxpy as cp
import sys
from collections import defaultdict
import pandas as pd
import math
import os

from suggest_utils import calc_reviewer_db_mapping, print_text_report, print_progress

BATCH_SIZE = 64
entok = MosesTokenizer(lang='en')

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
        print_progress(i, BATCH_SIZE)
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
        print_progress(i, BATCH_SIZE)
    print("", file=sys.stderr)
    return embeddings


def calc_similarity_matrix(model, db, quer):
    db_emb = create_embeddings(model, db)
    quer_emb = create_embeddings(model, quer)
    print(f'Performing similarity calculation', file=sys.stderr)
    return np.matmul(quer_emb, np.transpose(db_emb))


def calc_aggregate_reviewer_score(rdb, all_scores, operator='max'):
    """Calculate the aggregate reviewer score for one paper

    :param rdb: Reviewer DB. NP matrix of DB papers by reviewers
    :param scores: NP matrix of similarity scores between the current papers (rows) and the DB papers (columns)
    :param operator: Which operator to apply (max, weighted_topK)
    :return: Numpy matrix of length reviewers indicating the score for that reviewer
    """
    agg = np.zeros( (all_scores.shape[0], rdb.shape[1]) )
    print(f'Calculating aggregate scores for {all_scores.shape[0]} examples (.=10 examples)', file=sys.stderr)
    for i in range(all_scores.shape[0]):
        scores = all_scores[i]
        INVALID_SCORE = 0
        # slow -- 2-3 secs
        scored_rdb = rdb * scores.reshape((len(scores), 1)) + (1-rdb) * INVALID_SCORE
        if operator == 'max':
            agg[i] = np.amax(scored_rdb, axis=0)
        elif operator.startswith('weighted_top'):
            k = int(operator[12:])
            weighting = np.reshape(1/np.array(range(1, k+1)), (k,1))
            # slow -- 2-3 secs
            scored_rdb.sort(axis=0)
            topk = scored_rdb[-k:,:]
            agg[i] = (topk*weighting).sum(axis=0)
        else:
            raise ValueError(f'Unknown operator {operator}')
        print_progress(i, mod_size=10)
    print('', file=sys.stderr)
    return agg


def create_suggested_assignment(reviewer_scores, reviews_per_paper=3, min_papers_per_reviewer=0, max_papers_per_reviewer=5, quotas=None):
    num_pap, num_rev = reviewer_scores.shape
    if num_rev*max_papers_per_reviewer < num_pap*reviews_per_paper:
        raise ValueError(f'There are not enough reviewers ({num_rev}) review all the papers ({num_pap})'
                         f' given a constraint of {reviews_per_paper} reviews per paper and'
                         f' {max_papers_per_reviewer} reviews per reviewer')
    if num_rev*min_papers_per_reviewer > num_pap*reviews_per_paper:
        pass
        #raise ValueError(f'There are too many reviewers ({num_rev}) to review all the papers ({num_pap})'
        #                 f' given a constraint of {reviews_per_paper} reviews per paper and'
        #                 f' a minimum of {min_papers_per_reviewer} reviews per reviewer')
    assignment = cp.Variable(shape=reviewer_scores.shape, boolean=True)
    if not quotas:
        maxrev_constraint = cp.sum(assignment, axis=0) <= max_papers_per_reviewer
    else:
        max_papers = np.zeros((num_rev,), dtype=np.int32)
        max_papers[:] = max_papers_per_reviewer
        for j, q in quotas.items():
            max_papers[j] = min(q, max_papers_per_reviewer)
        maxrev_constraint = cp.sum(assignment, axis=0) <= max_papers
        
    pap_constraint = cp.sum(assignment, axis=1) == reviews_per_paper
    constraints = [maxrev_constraint, pap_constraint]
    if min_papers_per_reviewer > 0:
        if not quotas:
            minrev_constraint = cp.sum(assignment, axis=0) >= min_papers_per_reviewer
        else:
            min_papers = np.zeros((num_rev,), dtype=np.int32)
            min_papers[:] = min_papers_per_reviewer
            for j, q in quotas.items():
                min_papers[j] = min(min_papers_per_reviewer, q)
            minrev_constraint = cp.sum(assignment, axis=0) >= min_papers
        constraints.append(minrev_constraint)
    total_sim = cp.sum(cp.multiply(reviewer_scores, assignment))
    assign_prob = cp.Problem(cp.Maximize(total_sim), constraints)
    assign_prob.solve(solver=cp.GLPK_MI, verbose=True)
    return assignment.value, assign_prob.value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line JSON file of submissions")
    parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A json file of reviewer names and IDs that can review this time")
    parser.add_argument("--suggestion_file", type=str, required=True, help="An output file for the suggestions")
    parser.add_argument("--bid_file", type=str, default=None, help="A file containing numpy array of bids (0 = COI, 1 = no, 2 = maybe, 3 = yes)."+
                                                                   " Each row corresponds to a paper (line in submission_file) and each column corresponds to a reviewer (line in reviewer_file)."
                                                                   " This will be used to remove COIs, so just '0' and '3' is fine as well.")
    parser.add_argument("--filter_field", type=str, default="id", help="Which field to use as the reviewer ID (name/id)")
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--aggregator", type=str, default="weighted_top3", help="Aggregation type (max, weighted_topN where N is a number)")
    parser.add_argument("--paper_matrix", help="A filename for where to load/save the paper similarity matrix")
    parser.add_argument("--aggregate_matrix", help="A filename for where to load/save the reviewer-paper aggregate matrix")
    parser.add_argument("--max_papers_per_reviewer", default=20, type=int, help="How many papers, maximum, to assign to each reviewer")
    parser.add_argument("--min_papers_per_reviewer", default=10, type=int, help="How many papers, minimum, to assign to each reviewer")
    parser.add_argument("--reviews_per_paper", default=1, type=int, help="How many reviews to assign to each paper")
    parser.add_argument("--output_type", default="json", type=str, help="What format of output to produce (json/text)")

    parser.add_argument("--quota_file", help="A CSV file listing reviewer usernames with their maximum number of papers")

    args = parser.parse_args()

    # Load the data
    with open(args.submission_file, "r") as f:
        submissions = [json.loads(x) for x in f]
        submission_abs = [x['paperAbstract'] for x in submissions]
        submission_index = dict([(x['startSubmissionId'], x) for x in submissions])
    with open(args.reviewer_file, "r") as f:
        reviewer_data_orig = [json.loads(x) for x in f]
        reviewer_data = []
        reviewer_remapping = {}
        for i, data in enumerate(reviewer_data_orig):
            if data['areaChair']: 
                reviewer_remapping[i] = len(reviewer_data)
                reviewer_data.append(data)
            else:
                reviewer_remapping[i] = -1
        for data in reviewer_data:
            if 'name' in data:
                data['names'] = [data['name']]
                del data['name']
        reviewer_names = [x['names'][0] for x in reviewer_data]
        print(f'Have {len(reviewer_data)} reviewers', file=sys.stderr)
    with open(args.db_file, "r") as f:
        db = [json.loads(x) for x in f]  # for debug
        db_abs = [x['paperAbstract'] for x in db]
    rdb = calc_reviewer_db_mapping(reviewer_data, db, author_field='authors')
    
    # At least half of the above papers are not authored by reviewers, hacking them out
    includes_reviewer = rdb.sum(axis=1)
    new_db = []
    for i, paper in enumerate(db):
        if includes_reviewer[i] >= 1:
            new_db.append(paper)
    db = new_db
    db_abs = [x['paperAbstract'] for x in db]
    rdb = calc_reviewer_db_mapping(reviewer_data, db, author_field='authors')

    # Calculate or load paper similarity matrix
    if args.paper_matrix and os.path.exists(args.paper_matrix):
        mat = np.load(args.paper_matrix)
        assert(mat.shape[0] == len(submission_abs) and mat.shape[1] == len(db_abs))
    else:
        print('Loading model', file=sys.stderr)
        model, epoch = load_model(None, args.model_file, force_cpu=True)
        model.eval()
        assert not model.training
        mat = calc_similarity_matrix(model, db_abs, submission_abs)
        if args.paper_matrix:
            np.save(args.paper_matrix, mat)

    # Calculate reviewer scores based on paper similarity scores
    if args.aggregate_matrix and os.path.exists(args.aggregate_matrix):
        reviewer_scores = np.load(args.aggregate_matrix)
        assert(reviewer_scores.shape[0] == len(submission_abs) and reviewer_scores.shape[1] == len(reviewer_names))
    else:
        print('Calculating aggregate reviewer scores', file=sys.stderr)
        reviewer_scores = calc_aggregate_reviewer_score(rdb, mat, args.aggregator)
        if args.aggregate_matrix:
            np.save(args.aggregate_matrix, reviewer_scores)

    # Load and process COIs
    cois = np.where(np.load(args.bid_file) == 0, 1, 0) if args.bid_file else None
    if cois is not None:
        paper_rows, rev_cols = cois.nonzero()
        num_cois = 0
        for i in range(len(paper_rows)):
            pid, rid = paper_rows[i], rev_cols[i]
            acid = reviewer_remapping[rid]
            if acid != -1:
                reviewer_scores[pid,acid] = -1e5
                num_cois += 1
        print(f'Applying {num_cois} COIs', file=sys.stderr)

        #assert(cois.shape == reviewer_scores.shape)
        #num_cois = np.sum(cois)
        #print(f'Applying {num_cois} COIs', file=sys.stderr)
        #reviewer_scores = np.where(cois == 0, reviewer_scores, -1e5)

    # Load reviewer specific quotas
    quotas = {}
    username_to_idx = dict([(r['startUsername'], j) for j, r in enumerate(reviewer_data)])
    if args.quota_file:
        quotas_table = pd.read_csv(args.quota_file, skipinitialspace=True, quotechar='"', encoding = "UTF-8")
        quotas_table.fillna('', inplace=True)
        for i, line in quotas_table.iterrows():
            u, q = line['Username'], line['Quota']
            idx = username_to_idx.get(u)
            if idx != None:
                quotas[idx] = int(q)
            else:
                raise ValueError(f'Reviewer account {u} in quota file not found in reviewer database')
        print(f'Set {len(quotas)} reviewer quotas', file=sys.stderr)

    to_move = pd.read_csv('/Users/tcohn/software/emnlp.20200628/Moving-to-COI-Track.csv', skipinitialspace=True, quotechar='"', encoding = "UTF-8")
    to_move.fillna('', inplace=True)
    for i, line in to_move.iterrows():
        if line['Submission ID']: # skip blanks
            submission = submission_index[str(int(line['Submission ID']))]
            submission['track'] = line['Move To']

    # Ensure that reviewer tracks match the paper track
    # index the papers and reviewers by track
    track_papers = defaultdict(list)
    track_reviewers = defaultdict(list)
    for j, reviewer in enumerate(reviewer_data):
        track_reviewers[reviewer['track']].append(j)
    for i, submission in enumerate(submissions):
        if submission['track']:
            track_papers[submission['track']].append(i)
        else:
            raise ValueError(f'Submission {submission["startSubmissionId"]} has no track assigned')

    ptr = set(track_papers.keys())
    # if ptr.difference(track_reviewers.keys()):
    #     raise ValueError(f'Tracks mismatch between submissions and reviewers')
    if set(track_reviewers).difference(track_papers.keys()):
        print(f'WARNING: more reviewer tracks than papers "{set(track_reviewers).difference(track_papers.keys())}"')

    # Calculate a reviewer assignment based on the constraints
    assignment = np.zeros_like(reviewer_scores) 
    for track in sorted(track_papers.keys()):
        if track == 'Withdrawn': continue
        ps = track_papers[track]
        rs = track_reviewers[track]

        av_load = len(ps) / len(rs) * args.reviews_per_paper
        #min_papers = min(args.min_papers_per_reviewer, int(av_load * 0.85))
        min_papers = int(av_load * 0.85)
        #max_papers = max(min(args.max_papers_per_reviewer, int(av_load / 0.85)), math.ceil(av_load))
        max_papers = max(int(av_load / 0.85), math.ceil(av_load))

        track_rev_scores = reviewer_scores[ps,:][:,rs]
        track_quotas = dict([(rs.index(rid), rq) for rid, rq in quotas.items() if rid in rs])
        print(f'Calculating assignment of reviewers to track {track}', file=sys.stderr)
        print(f'\t{len(ps)} papers {len(rs)} reviewers', file=sys.stderr)
        print(f'\tmin load {min_papers} max load {max_papers} quotas {len(track_quotas)}', file=sys.stderr)
        track_assignment, assignment_score = create_suggested_assignment(track_rev_scores,
                                                min_papers_per_reviewer=min_papers,
                                                max_papers_per_reviewer=max_papers,
                                                reviews_per_paper=args.reviews_per_paper, quotas=track_quotas)
        print(f'Done calculating assignment, total score: {assignment_score}', file=sys.stderr)

        for pid, rid in zip(*track_assignment.nonzero()):
            assignment[ps[pid], rs[rid]] = 1

    # Print out the results
    with open(args.suggestion_file, 'w') as outf:
        for i, query in enumerate(submissions):
            scores = mat[i]
            best_idxs = scores.argsort()[-5:][::-1]
            best_reviewers = reviewer_scores[i].argsort()[-5:][::-1]
            assigned_reviewers = assignment[i].argsort()[-args.reviews_per_paper:][::-1]

            ret_dict = dict(query)
            ret_dict['similarPapers'] = [{'title': db[idx]['title'], 'paperAbstract': db[idx]['paperAbstract'], 'score': scores[idx]} for idx in best_idxs]
            ret_dict['topSimReviewers'], ret_dict['assignedReviewers'] = [], []
            for idx in best_reviewers:
                next_dict = dict(reviewer_data[idx])
                next_dict['score'] = reviewer_scores[i][idx]
                ret_dict['topSimReviewers'].append(next_dict)
            for idx in assigned_reviewers:
                next_dict = dict(reviewer_data[idx])
                next_dict['score'] = reviewer_scores[i][idx]
                ret_dict['assignedReviewers'].append(next_dict)

            if args.output_type == 'json':
                print(json.dumps(ret_dict), file=outf)
            elif args.output_type == 'text':
                print_text_report(ret_dict, file=outf)
            else:
                raise ValueError(f'Illegal output_type {args.output_type}')

    print(f'Done creating suggestions, written to {args.suggestion_file}', file=sys.stderr)

