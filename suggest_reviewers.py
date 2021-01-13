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

from suggest_utils import calc_reviewer_db_mapping, print_text_report, print_progress

BATCH_SIZE = 128
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


def create_suggested_assignment(reviewer_scores, reviews_per_paper=3, min_papers_per_reviewer=0,
                                max_papers_per_reviewer=5, quotas=None, anonymity_multiplier=1.0):
    """Create a suggested reviewer assignment

    :param reviewer_scores: The similarity scores used to assign reviewers
    :param reviews_per_paper: Maximum number of reviews per paper
    :param min_papers_per_reviewer: Minimum number of reviews per paper
    :param max_papers_per_reviewer: Maximum number of papers per reviewer
    :param quotas: Per-reviewer quota on the maximum number of papers
    :param anonymity_multiplier: A multiplier in how many extra papers to assign (for anonymization purposes)
    :return: An assignment of papers to reviewers, and a score indicating the quality of the assignment
    """
    num_pap, num_rev = reviewer_scores.shape
    if num_rev*max_papers_per_reviewer < num_pap*reviews_per_paper:
        raise ValueError(f'There are not enough reviewers ({num_rev}) review all the papers ({num_pap})'
                         f' given a constraint of {reviews_per_paper} reviews per paper and'
                         f' {max_papers_per_reviewer} reviews per reviewer')
    if num_rev*min_papers_per_reviewer > num_pap*reviews_per_paper:
        raise ValueError(f'There are too many reviewers ({num_rev}) to review all the papers ({num_pap})'
                         f' given a constraint of {reviews_per_paper} reviews per paper and'
                         f' a minimum of {min_papers_per_reviewer} reviews per reviewer')
    if anonymity_multiplier < 1.0:
        raise ValueError(f'anonymity_multiplier must be >= 1.0')
    assignment = cp.Variable(shape=reviewer_scores.shape, boolean=True)
    if not quotas:
        maxrev_constraint = cp.sum(assignment, axis=0) <= max_papers_per_reviewer*anonymity_multiplier
    else:
        max_papers = np.zeros((num_rev,), dtype=np.int32)
        max_papers[:] = max_papers_per_reviewer
        for j, q in quotas.items():
            max_papers[j] = q
            if q > max_papers_per_reviewer:
                print(f'WARNING setting max_papers to {q} exceeds default value of {max_papers_per_reviewer}')
        maxrev_constraint = cp.sum(assignment, axis=0) <= max_papers*anonymity_multiplier
        
    pap_constraint = cp.sum(assignment, axis=1) == int(reviews_per_paper*anonymity_multiplier)
    constraints = [maxrev_constraint, pap_constraint]
    if min_papers_per_reviewer > 0:
        if not quotas:
            minrev_constraint = cp.sum(assignment, axis=0) >= min_papers_per_reviewer*anonymity_multiplier
        else:
            min_papers = np.zeros((num_rev,), dtype=np.int32)
            min_papers[:] = min_papers_per_reviewer
            for j, q in quotas.items():
                min_papers[j] = min(min_papers_per_reviewer, q)
            minrev_constraint = cp.sum(assignment, axis=0) >= min_papers*anonymity_multiplier
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
    parser.add_argument("--filter_field", type=str, default="name", help="Which field to use as the reviewer ID (name/id)")
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--aggregator", type=str, default="weighted_top3", help="Aggregation type (max, weighted_topN where N is a number)")
    parser.add_argument("--save_paper_matrix", help="A filename for where to save the paper similarity matrix")
    parser.add_argument("--load_paper_matrix", help="A filename for where to load the cached paper similarity matrix")
    parser.add_argument("--save_aggregate_matrix", help="A filename for where to save the reviewer-paper aggregate matrix")
    parser.add_argument("--load_aggregate_matrix", help="A filename for where to load the cached reviewer-paper aggregate matrix")
    parser.add_argument("--max_papers_per_reviewer", default=5, type=int, help="How many papers, maximum, to assign to each reviewer")
    parser.add_argument("--min_papers_per_reviewer", default=0, type=int, help="How many papers, minimum, to assign to each reviewer")
    parser.add_argument("--reviews_per_paper", default=3, type=int, help="How many reviews to assign to each paper")
    parser.add_argument("--anonymity_multiplier", default=1.0, type=float,
                            help="For anonymity purposes, it is possible to assign extra reviewers to papers and then "
                                 "sub-sample these reviewers. Set to, for example, 2.0 to assign an initial review committee"
                                 "twice the normal size, then sub-sample down to the desired size.")
    parser.add_argument("--output_type", default="json", type=str, help="What format of output to produce (json/text)")

    parser.add_argument("--quota_file", help="A CSV file listing reviewer usernames with their maximum number of papers")
    parser.add_argument("--track", action='store_true', help="Ensure reviewers and papers match in terms of track")
    parser.add_argument("--area_chairs", action='store_true', help="Assign papers to area chairs (default is reviewers); ensure min/max_papers_per_reviewer are set accordingly")
    #parser.add_argument("--short_paper_weight", type=float, default=0.7, help="How to count a short paper relative to a long paper when assessing quota")

    args = parser.parse_args()

    # Load the data
    with open(args.submission_file, "r") as f:
        submissions = [json.loads(x) for x in f]
        submission_abs = [x['paperAbstract'] for x in submissions]
    with open(args.reviewer_file, "r") as f:
        reviewer_data = [json.loads(x) for x in f]
        for data in reviewer_data:
            if 'name' in data:
                data['names'] = [data['name']]
                del data['name']
        reviewer_names = [x['names'][0] for x in reviewer_data]
    with open(args.db_file, "r") as f:
        db = [json.loads(x) for x in f]  # for debug
        db_abs = [x['paperAbstract'] for x in db]
    rdb = calc_reviewer_db_mapping(reviewer_data, db, author_field='authors')
    
    # FIXME: about half of the above papers are bollocks -- quick hack to filter to those papers actually 
    # authored by reviewers
    includes_reviewer = rdb.sum(axis=1)
    new_db = []
    for i, paper in enumerate(db):
        if includes_reviewer[i] >= 1:
            new_db.append(paper)
    db = new_db
    db_abs = [x['paperAbstract'] for x in db]
    rdb = calc_reviewer_db_mapping(reviewer_data, db, author_field='authors')

    # Calculate or load paper similarity matrix
    if args.load_paper_matrix:
        mat = np.load(args.load_paper_matrix)
        assert(mat.shape[0] == len(submission_abs) and mat.shape[1] == len(db_abs))
    else:
        print('Loading model', file=sys.stderr)
        model, epoch = load_model(None, args.model_file, force_cpu=True)
        model.eval()
        assert not model.training
        mat = calc_similarity_matrix(model, db_abs, submission_abs)
        if args.save_paper_matrix:
            np.save(args.save_paper_matrix, mat)

    # Calculate reviewer scores based on paper similarity scores
    if args.load_aggregate_matrix:
        reviewer_scores = np.load(args.load_aggregate_matrix)
        assert(reviewer_scores.shape[0] == len(submission_abs) and reviewer_scores.shape[1] == len(reviewer_names))
    else:
        print('Calculating aggregate reviewer scores', file=sys.stderr)
        reviewer_scores = calc_aggregate_reviewer_score(rdb, mat, args.aggregator)
        if args.save_aggregate_matrix:
            np.save(args.save_aggregate_matrix, reviewer_scores)

    # Load and process COIs
    cois = np.where(np.load(args.bid_file) == 0, 1, 0) if args.bid_file else None
    if cois is not None:
        num_cois = np.sum(cois)
        print(f'Applying {num_cois} COIs', file=sys.stderr)
        reviewer_scores = np.where(cois == 0, reviewer_scores, -1e5)

    # Load reviewer specific quotas
    quotas = {}
    if args.quota_file:
        username_to_idx = dict([(r['startUsername'], j) for j, r in enumerate(reviewer_data)])
    
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

    # Ensure ACs are excluded from assignment, unless --area_chairs option specified
    num_excluded = num_included = 0
    for j, reviewer in enumerate(reviewer_data):
        if args.area_chairs:
            if not reviewer.get('areaChair', False):
                num_excluded += 1
                reviewer_scores[:,j] = -1e5
            else:
                num_included += 1
        else:
            if reviewer.get('seniorAreaChair', False) or reviewer.get('areaChair', False):
                num_excluded += 1
                reviewer_scores[:,j] = -1e5
            else:
                num_included += 1
    print(f'Excluded {num_excluded} reviewers/chairs, leaving {num_included}', file=sys.stderr)
                
    # Ensure that reviewer tracks match the paper track
    if args.track:
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
        ptr.add('Multidisciplinary and AC COI')
        rtr = set(track_reviewers.keys())
#        if track_papers.keys() != track_reviewers.keys():
#            raise ValueError(f'Tracks mismatch between submissions and reviewers')

        # mask defines valid reviewer paper pairings
        mask = np.zeros_like(reviewer_scores)
        for track in track_papers.keys():
            ps = track_papers[track]
            rs = track_reviewers[track]
            for p in ps:
                for r in rs:
                    mask[p,r] = 1

        reviewer_scores = np.where(mask == 1, reviewer_scores, -1e5)
        print(f'Applying track constraints for {len(rtr)} tracks', file=sys.stderr)

    # FIXME: should we weight short papers separately to long papers in the review assignments?
    # E.g., assume 5 long papers = 7 short papers, or is this too painful

    # Calculate a reviewer assignment based on the constraints
    final_scores = reviewer_scores
    if args.anonymity_multiplier != 1.0:
        print('Calculating initial assignment of reviewers', file=sys.stderr)
        final_scores, assignment_score = create_suggested_assignment(reviewer_scores,
                                                                     min_papers_per_reviewer=args.min_papers_per_reviewer,
                                                                     max_papers_per_reviewer=args.max_papers_per_reviewer,
                                                                     reviews_per_paper=args.reviews_per_paper,
                                                                     quotas=quotas,
                                                                     anonymity_multiplier=args.anonymity_multiplier)
        print(f'Done calculating initial assignment, total score: {assignment_score}', file=sys.stderr)
        final_scores += np.random.random(final_scores.shape)*1e-4
    print('Calculating assignment of reviewers', file=sys.stderr)
    assignment, assignment_score = create_suggested_assignment(final_scores,
                                             min_papers_per_reviewer=args.min_papers_per_reviewer,
                                             max_papers_per_reviewer=args.max_papers_per_reviewer,
                                             reviews_per_paper=args.reviews_per_paper, quotas=quotas)
    print(f'Done calculating assignment, total score: {assignment_score}', file=sys.stderr)

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

