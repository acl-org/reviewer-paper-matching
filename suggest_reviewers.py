import argparse
import csv
import json
import os
import re
import sys
import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
from collections import defaultdict
from sacremoses import MosesTokenizer
from model_utils import Example, unk_string
from models import load_model
from suggest_utils import (
    calc_reviewer_db_mapping, print_progress, print_text_report
)

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')


def assign_quotas(
    reviewers,
    quota_file,
    max_papers,
    area_chairs=False,
):
    """
    TODO: add a docstring
    """
    quotas = {}
    username_to_idx = dict(
        [(r['startUsername'], j) for j, r in enumerate(reviewers)]
    )
    quota_table = pd.read_csv(
        quota_file, skipinitialspace=True, quotechar='"', encoding="UTF-8"
    )

    # Quota file will usually be a csv downloaded from the Author/Reviewer
    # Profiles section of the softconf spreadsheet maker. Thus it will
    # include both authors and reviewers, and only the reviewers are
    # relevant. Here, filter only the reviewers, fill any nan values with
    # the default reviewer quota, and replace the string value 'None' with
    # 0
    reviewer_usernames = set(username_to_idx.keys())
    quota_usernames = list(quota_table['Username'])
    quota_overlap = [name in reviewer_usernames for name in quota_usernames]
    quota_table = quota_table[quota_overlap].reset_index(drop=True)
    quota_table['QuotaForReview'].fillna(max_papers, inplace=True)
    quota_table['QuotaForReview'].replace(
        to_replace='None', value=0, inplace=True
    )

    for i, line in quota_table.iterrows():
        u, q = line['Username'], line['QuotaForReview']
        idx = username_to_idx.get(u)
        if area_chairs:
            num_tracks = len(reviewers[idx]['ac_tracks'])
        else:
            num_tracks = len(reviewers[idx]['tracks'])
        if idx != None:
            if num_tracks > 0:
                quotas[idx] = int(q) // num_tracks
            else:
                quotas[idx] = int(q)
        else:
            raise ValueError(
                f"Reviewer account {u} in quota file not found in reviewer"
                " database"
            )

    return quotas


def exclude_positions(reviewers, reviewer_scores, area_chairs=False):
    """
    TODO: add a docstring
    """
    num_excluded = num_included = 0
    for j, reviewer in enumerate(reviewers):
        if area_chairs:
            # only ACs will get paper assignments:
            ac = reviewer.get('areaChair', False)
            if not ac:
                # the reviewer is not AC and is excluded
                num_excluded += 1
                reviewer_scores[:, j] -= 150
            else:
                # the reviewer is an AC and is included
                num_included += 1
        else:
            # SACs and ACs will not get assignments
            sac = reviewer.get('seniorAreaChair', False)
            ac = reviewer.get('areaChair', False)
            if sac or ac:
                # the reviewer is an SAC or AC and is excluded
                num_excluded += 1
                reviewer_scores[:, j] -= 150
            else:
                # the reviewer is a reviewer and is included
                num_included += 1

    return reviewer_scores, num_included, num_excluded


def split_by_subproblem(
    reviewers,
    submissions,
    reviewer_scores,
    quotas=None,
    by_track=False,
    area_chairs=False
):
    """
    TODO: add a docstring
    """
    optimization_problems = {}
    problem_papers = defaultdict(list)
    problem_reviewers = defaultdict(list)
    problem_quotas = defaultdict(dict)

    if by_track:
        # Index the papers and reviewers by track
        for i, reviewer in enumerate(reviewers):
            if area_chairs:
                track_list = reviewer['ac_tracks']
            else:
                track_list = reviewer['tracks']
            if len(track_list) > 1:
                warnings.warn(
                    f"{reviewer['names'][0]} is in multiple tracks", UserWarning
                )
            for track in track_list:
                local_index = len(problem_reviewers[track])
                problem_reviewers[track].append(i)
                if quotas:
                    problem_quotas[track][local_index] = quotas[i]
        for i, submission in enumerate(submissions):
            if submission['track']:
                problem_papers[submission['track']].append(i)
            else:
                raise ValueError(
                    f"Submission {submission['startSubmissionId']} has no track"
                    " assigned"
                )
    else:
        for i, reviewer in enumerate(reviewers):
            if area_chairs:
                if reviewer['areaChair']:
                    local_index = len(problem_reviewers['all_tracks'])
                    problem_reviewers['all_tracks'].append(i)
                    if quotas:
                        problem_quotas['all_tracks'][local_index] = quotas[i]
            else:
                if not (reviewer['areaChair'] or reviewer['seniorAreaChair']):
                    local_index = len(problem_reviewers['all_tracks'])
                    problem_reviewers['all_tracks'].append(i)
                    if quotas:
                        problem_quotas['all_tracks'][local_index] = quotas[i]
        problem_papers['all_tracks'] = [i for i in range(len(submissions))]

    for problem in problem_papers.keys():
        problem_matrix = reviewer_scores[problem_papers[problem], :]
        problem_matrix = problem_matrix[:, problem_reviewers[problem]]
        optimization_problems[problem] = problem_matrix

    # FIXME: should we weight short papers separately to long papers in the
    # review assignments? E.g., assume 5 long papers = 7 short papers, or is
    # this too painful

    return (
        optimization_problems, problem_papers, problem_reviewers, problem_quotas
    )


def create_embeddings(model, examps):
    """
    Embed textual examples

    :param examps: A list of text to embed
    :return: A len(examps) by embedding size numpy matrix of embeddings
    """
    # Preprocess examples
    print(
        f'Preprocessing {len(examps)} examples (.={BATCH_SIZE} examples)',
        file=sys.stderr
    )
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
    print(
        f'Embedding {len(examps)} examples (.={BATCH_SIZE} examples)',
        file=sys.stderr
    )
    embeddings = np.zeros((len(examps), model.args.dim))
    for i in range(0, len(data), BATCH_SIZE):
        max_idx = min(i + BATCH_SIZE, len(data))
        curr_batch = data[i:max_idx]
        wx1, wl1 = model.torchify_batch(curr_batch)
        vecs = model.encode(wx1, wl1)
        vecs = vecs.detach().cpu().numpy()
        # Normalize for NN search
        vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None]
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
    """
    Calculate the aggregate reviewer score for one paper

    :param rdb: Reviewer DB. NP matrix of DB papers by reviewers
    :param scores: NP matrix of similarity scores between the current papers 
        (rows) and the DB papers (columns)
    :param operator: Which operator to apply (max, weighted_topK)
    :return: Numpy matrix of length reviewers indicating the score for that
        reviewer
    """
    agg = np.zeros((all_scores.shape[0], rdb.shape[1]))
    print(
        f'Calculating aggregate scores for {all_scores.shape[0]}'
        ' examples (.=10 examples)',
        file=sys.stderr
    )
    for i in range(all_scores.shape[0]):
        scores = all_scores[i]
        INVALID_SCORE = 0
        # slow -- 2-3 secs
        scored_rdb = rdb * scores.reshape((len(scores), 1)
                                         ) + (1 - rdb) * INVALID_SCORE
        if operator == 'max':
            agg[i] = np.amax(scored_rdb, axis=0)
        elif operator.startswith('weighted_top'):
            k = int(operator[12:])
            weighting = np.reshape(1 / np.array(range(1, k + 1)), (k, 1))
            # slow -- 2-3 secs
            scored_rdb.sort(axis=0)
            topk = scored_rdb[-k:, :]
            agg[i] = (topk * weighting).sum(axis=0)
        else:
            raise ValueError(f'Unknown operator {operator}')
        print_progress(i, mod_size=10)
    print('', file=sys.stderr)
    return agg


def create_suggested_assignment(
    reviewer_scores,
    reviews_per_paper=3,
    min_papers_per_reviewer=0,
    max_papers_per_reviewer=5,
    quotas=None,
    anonymity_multiplier=1.0
):
    """
    Create a suggested reviewer assignment

    :param reviewer_scores: The similarity scores used to assign reviewers
    :param reviews_per_paper: Maximum number of reviews per paper
    :param min_papers_per_reviewer: Minimum number of reviews per paper
    :param max_papers_per_reviewer: Maximum number of papers per reviewer
    :param quotas: Per-reviewer quota on the maximum number of papers
    :param anonymity_multiplier: A multiplier in how many extra papers to assign
        (for anonymization purposes)
    :return: An assignment of papers to reviewers, and a score indicating the
        quality of the assignment
    """
    num_pap, num_rev = reviewer_scores.shape
    if num_rev * max_papers_per_reviewer < num_pap * reviews_per_paper:
        raise ValueError(
            f"There are not enough reviewers ({num_rev}) review all the papers"
            f" ({num_pap}) given a constraint of {reviews_per_paper} reviews"
            f" per paper and {max_papers_per_reviewer} reviews per reviewer"
        )
    if num_rev * min_papers_per_reviewer > num_pap * reviews_per_paper:
        raise ValueError(
            f"There are too many reviewers ({num_rev}) to review all the papers"
            f" ({num_pap}) given a constraint of {reviews_per_paper} reviews"
            f" per paper and a minimum of {min_papers_per_reviewer} reviews per"
            " reviewer"
        )
    if anonymity_multiplier < 1.0:
        raise ValueError(f"anonymity_multiplier must be >= 1.0")
    assignment = cp.Variable(shape=reviewer_scores.shape, boolean=True)
    if not quotas:
        maxrev_constraint = cp.sum(
            assignment, axis=0
        ) <= max_papers_per_reviewer * anonymity_multiplier
    else:
        max_papers = np.zeros((num_rev, ), dtype=np.int32)
        max_papers[:] = max_papers_per_reviewer
        for j, q in quotas.items():
            max_papers[j] = q
            if q > max_papers_per_reviewer:
                print(
                    f"WARNING setting max_papers to {q} exceeds default value"
                    f" of {max_papers_per_reviewer}"
                )
        maxrev_constraint = cp.sum(
            assignment, axis=0
        ) <= max_papers * anonymity_multiplier

    pap_constraint = cp.sum(assignment, axis=1) == int(
        reviews_per_paper * anonymity_multiplier
    )
    constraints = [maxrev_constraint, pap_constraint]
    if min_papers_per_reviewer > 0:
        if not quotas:
            minrev_constraint = cp.sum(
                assignment, axis=0
            ) >= min_papers_per_reviewer * anonymity_multiplier
        else:
            min_papers = np.zeros((num_rev, ), dtype=np.int32)
            min_papers[:] = min_papers_per_reviewer
            for j, q in quotas.items():
                min_papers[j] = min(min_papers_per_reviewer, q)
            minrev_constraint = cp.sum(
                assignment, axis=0
            ) >= min_papers * anonymity_multiplier
        constraints.append(minrev_constraint)
    total_sim = cp.sum(cp.multiply(reviewer_scores, assignment))
    assign_prob = cp.Problem(cp.Maximize(total_sim), constraints)
    assign_prob.solve(solver=cp.GLPK_MI, verbose=True)
    return assignment.value, assign_prob.value


def main():
    """
    TODO: add a docstring
    """

    # --------------------------------------------------------------------------
    # Part 1: Read in the arguments
    # --------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--submission_file',
        type=str,
        required=True,
        help="A line-by-line JSON file of submissions"
    )
    parser.add_argument(
        '--db_file',
        type=str,
        required=True,
        help="File (in s2 json format) of relevant papers from reviewers"
    )
    parser.add_argument(
        '--reviewer_file',
        type=str,
        required=True,
        help="A json file of reviewer names and IDs that can review this time"
    )
    parser.add_argument(
        '--suggestion_file',
        type=str,
        required=True,
        help="An output file for the suggestions"
    )
    parser.add_argument(
        '--bid_file',
        type=str,
        default=None,
        help="A file containing numpy array of bids"
        " (0 = COI, 1 = no, 2 = maybe, 3 = yes)." +
        " Each row corresponds to a paper (line in submission_file) and each"
        " column corresponds to a reviewer (line in reviewer_file). This will"
        " be used to remove COIs, so just '0' and '3' is fine as well."
    )
    parser.add_argument(
        '--filter_field',
        type=str,
        default="name",
        help="Which field to use as the reviewer ID (name/id)"
    )
    parser.add_argument(
        '--model_file',
        help="filename to load the pre-trained semantic similarity file."
    )
    parser.add_argument(
        '--aggregator',
        type=str,
        default="weighted_top3",
        help="Aggregation type (max, weighted_topN where N is a number)"
    )
    parser.add_argument(
        '--save_paper_matrix',
        help="A filename for where to save the paper similarity matrix"
    )
    parser.add_argument(
        '--load_paper_matrix',
        help="A filename for where to load the cached paper similarity matrix"
    )
    parser.add_argument(
        '--save_aggregate_matrix',
        help="A filename for where to save the reviewer-paper aggregate matrix"
    )
    parser.add_argument(
        '--load_aggregate_matrix',
        help="A filename for where to load the cached reviewer-paper aggregate"
        " matrix"
    )
    parser.add_argument(
        '--save_assignment_matrix',
        help="A filename for where to sae the assignment matrix"
    )
    parser.add_argument(
        '--load_assignment_matrix',
        help="A filename for wher to load the cached assignment matrix"
    )
    parser.add_argument(
        '--max_papers_per_reviewer',
        default=5,
        type=int,
        help="How many papers, maximum, to assign to each reviewer"
    )
    parser.add_argument(
        '--min_papers_per_reviewer',
        default=0,
        type=int,
        help="How many papers, minimum, to assign to each reviewer"
    )
    parser.add_argument(
        '--reviews_per_paper',
        default=3,
        type=int,
        help="How many reviews to assign to each paper"
    )
    parser.add_argument(
        '--num_similar_to_list',
        default=3,
        type=int,
        help="How many similar reviewers to list in addition to the assigned"
        " ones"
    )
    parser.add_argument(
        '--anonymity_multiplier',
        default=1.0,
        type=float,
        help="For anonymity purposes, it is possible to assign extra reviewers"
        " to papers and then sub-sample these reviewers. Set to, for example,"
        " 2.0 to assign an initial review committee twice the normal size, then"
        " sub-sample down to the desired size."
    )
    parser.add_argument(
        '--output_type',
        default="json",
        type=str,
        help="What format of output to produce (json/text)"
    )
    parser.add_argument(
        '--quota_file',
        help="A CSV file listing reviewer usernames with their maximum number"
        " of papers"
    )
    parser.add_argument(
        '--assignment_spreadsheet',
        help="A filepath to which to write global and track-wise human-readable"
        " assignment spreadsheets"
    )
    parser.add_argument(
        '--track',
        action='store_true',
        help="Ensure reviewers and papers match in terms of track"
    )
    parser.add_argument(
        '--area_chairs',
        action='store_true',
        help="Assign papers to area chairs (default is reviewers); ensure"
        " min/max_papers_per_reviewer are set accordingly"
    )
    #parser.add_argument(
    #    "--short_paper_weight",
    #    type=float, default=0.7,
    #    help=
    #    "How to count a short paper relative to a long paper when assessing"
    #    " quota"
    #)

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Part 2: Load the data and calculate similarity between submissions and
    # reviewers
    # --------------------------------------------------------------------------

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

    # FIXME: about half of the above papers are bollocks -- quick hack to filter
    # to those papers actually authored by reviewers
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
        assert (
            mat.shape[0] == len(submission_abs) and mat.shape[1] == len(db_abs)
        )
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
        assert (
            reviewer_scores.shape[0] == len(submission_abs) and
            reviewer_scores.shape[1] == len(reviewer_names)
        )
    else:
        print('Calculating aggregate reviewer scores', file=sys.stderr)
        reviewer_scores = calc_aggregate_reviewer_score(
            rdb, mat, args.aggregator
        )
        if args.save_aggregate_matrix:
            np.save(args.save_aggregate_matrix, reviewer_scores)

    # --------------------------------------------------------------------------
    # Part 3: Adjust reviewer_scores based on COI, AC role; add quota
    # constraints; optionally split into subproblems by track
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Part 3(a): Adjust reviewer_scores based on COIs
    # --------------------------------------------------------------------------

    cois = np.where(
        np.load(args.bid_file) == 0, 1, 0
    ) if args.bid_file else None
    if cois is not None:
        num_cois = np.sum(cois)
        print(f"Applying {num_cois} COIs", file=sys.stderr)
        reviewer_scores = np.where(
            cois == 0, reviewer_scores, reviewer_scores - 110
        )

    # --------------------------------------------------------------------------
    # Part 3(b): Load reviewer specific quotas
    # --------------------------------------------------------------------------

    quotas = {}
    if args.quota_file:
        quotas = assign_quotas(
            reviewer_data,
            args.quota_file,
            args.max_papers_per_reviewer,
            area_chairs=args.area_chairs
        )
    print(f"Set {len(quotas)} reviewer quotas", file=sys.stderr)

    # --------------------------------------------------------------------------
    # Part 3(c): Adjust reviewer_scores based on ACs
    # If --area_chairs is specified, only ACs get papers. If it is not
    # specified, SACs and ACs should not get papers or be shown as similar
    # reviewers (i.e., the reviewer_score is set to -150 for those positions)
    # --------------------------------------------------------------------------

    reviewer_scores, num_included, num_excluded = exclude_positions(
        reviewer_data, reviewer_scores, area_chairs=args.area_chairs
    )
    print(
        f"Excluded {num_excluded} reviewers/chairs, leaving {num_included}",
        file=sys.stderr
    )

    # --------------------------------------------------------------------------
    # Part 4: Break the optimization into subproblems
    # If --track is not specified, there will be a single subproblem called
    # ``all_tracks``, although ACs or reviewers will be excluded as necessary.
    # If --track is specified, the matrix will be broken into one optimization
    # subproblem per track
    # --------------------------------------------------------------------------

    optimization_problems, problem_papers, problem_reviewers, problem_quotas = (
        split_by_subproblem(
            reviewer_data,
            submissions,
            reviewer_scores,
            quotas,
            by_track=args.track,
            area_chairs=args.area_chairs
        )
    )

    # --------------------------------------------------------------------------
    # Part 5: Calculate a reviewer assignment based on the constraints
    # --------------------------------------------------------------------------
    assignments = {}
    assignment_scores = {}
    for problem in optimization_problems.keys():
        final_scores = optimization_problems[problem]

        if args.anonymity_multiplier != 1.0:
            print(
                "Calculating initial assignment of reviewers for category"
                f" {problem}",
                file=sys.stderr
            )
            final_scores, assignment_score = create_suggested_assignment(
                final_scores,
                min_papers_per_reviewer=args.min_papers_per_reviewer,
                max_papers_per_reviewer=args.max_papers_per_reviewer,
                reviews_per_paper=args.reviews_per_paper,
                quotas=problem_quotas[problem],
                anonymity_multiplier=args.anonymity_multiplier
            )
            print(
                "Done calculating initial assignment,"
                f" total score: {assignment_score}",
                file=sys.stderr
            )
            final_scores += np.random.random(final_scores.shape) * 1e-4

        print(
            f"Calculating assignment of reviewers for category {problem}",
            file=sys.stderr
        )

        # final_scores includes the penalties for COI. The constraints for CP
        # itself are only the quota constraints (max/min # of papers a reviewer
        # wants to review)
        assignment, assignment_score = create_suggested_assignment(
            final_scores,
            min_papers_per_reviewer=args.min_papers_per_reviewer,
            max_papers_per_reviewer=args.max_papers_per_reviewer,
            reviews_per_paper=args.reviews_per_paper,
            quotas=problem_quotas[problem]
        )

        assignments[problem] = assignment
        assignment_scores[problem] = assignment_score

        print(
            f"Done calculating assignment. Total score: {assignment_score}",
            file=sys.stderr
        )
        if assignment is None:
            warnings.warn(
                f"No solution found for category {problem}", RuntimeWarning
            )

    # --------------------------------------------------------------------------
    # Part 6: Print out the results in jsonl format
    # --------------------------------------------------------------------------

    with open(args.suggestion_file, 'w') as outf:
        problem = 'all_tracks'
        for i, submission in enumerate(submissions):
            track = submission['track']
            if args.track:
                problem = track
            category_idx = problem_papers[problem].index(i)
            scores = mat[i]
            best_idxs = scores.argsort()[-args.num_similar_to_list:][::-1]
            best_reviewers = (reviewer_scores[i].argsort()[-args.num_similar_to_list:][::-1])
            try:
                assigned_reviewers = (
                    assignments[problem][category_idx].argsort()
                    [-args.reviews_per_paper:][::-1]
                )
            except:
                assigned_reviewers = []
            ret_dict = dict(submissions[i])
            ret_dict['similarPapers'] = [
                {
                    'title': db[idx]['title'],
                    'paperAbstract': db[idx]['paperAbstract'],
                    'score': scores[idx]
                } for idx in best_idxs
            ]
            ret_dict['topSimReviewers'] = []
            ret_dict['assignedReviewers'] = []
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

    print(
        f"Done creating suggestions, written to {args.suggestion_file}\n",
        file=sys.stderr
    )

    # --------------------------------------------------------------------------
    # Part 7 (optional): Print out the results in more human-readable
    # spreadsheets
    # --------------------------------------------------------------------------

    # ACL-2021: We are outputting an alternative data file to easily create
    # a per-track spreadsheet of assigned reviewers, as well as a global file
    # with the minimum assignment information
    if args.assignment_spreadsheet:

        # Create a list of submissions both by track and globally
        track_assignments = defaultdict(list)
        global_assignments = []

        # Form the csv headers such that both spreadsheets will contain the id
        # of the submission and the username of each assigned reviewer/AC
        track_header_info = ['ID']
        global_header_info = ['ID']
        for x in range(1, args.reviews_per_paper + 1):
            if args.area_chairs:
                ac_field = f'Assigned AC {x}'
                score_field = f'Assigned AC {x} score'
                track_header_info += [ac_field, score_field]
                global_header_info += [ac_field, score_field]
            else:
                reviewer_field = f'Assigned reviewer {x}'
                score_field = f'Assigned reviewer {x} score'
                track_header_info += [reviewer_field, score_field]
                global_header_info += [reviewer_field, score_field]
        for x in range(1, args.num_similar_to_list + 1):
            if args.area_chairs:
                ac_field = f'Similar AC {x}'
                score_field = f'Similar AC {x} score'
                track_header_info += [ac_field, score_field]
            else:
                reviewer_field = f'Similar reviewer {x}'
                score_field = f'Similar reviewer {x} score'
                track_header_info += [reviewer_field, score_field]

        global_header_info += ['Track', 'Assigned within same track']

        # The by-track spreadsheet will also show SACs and ACs with COIs
        track_header_info += ['SACs with COI', 'ACs with COI']

        problem = 'all_tracks'

        # Loop over the submissions
        for submission_id, submission in enumerate(submissions):

            # Get the track and submission id for each
            track_submission_info = []
            global_submission_info = []
            track = submission['track']
            if args.track:
                problem = track
            submission_local_id = problem_papers[problem].index(submission_id)
            submission_idx = submission['startSubmissionId']
            track_submission_info.append(submission_idx)
            global_submission_info.append(submission_idx)

            # Use the scores to get the index for each assigned reviewer, and
            # get their username for the output spreadsheet
            try:
                assigned_reviewers = (
                    assignments[problem][submission_local_id].argsort()
                    [-args.reviews_per_paper:][::-1]
                )
            except:
                assigned_reviewers = []

            assigned_reviewer_global_ids = [
                problem_reviewers[problem][j] for j in assigned_reviewers
            ]
            similar_reviewers = (
                optimization_problems[problem][submission_local_id].argsort()
                [-args.num_similar_to_list:][::-1]
            )
            similar_reviewer_global_ids = [
                problem_reviewers[problem][j] for j in similar_reviewers
            ]

            same_track = True
            if len(assigned_reviewers) == 0:
                for i in range(args.reviews_per_paper):
                    track_submission_info += ['', '']
                    global_submission_info += ['', '']
            else:
                reviewer_ids = zip(
                    assigned_reviewers, assigned_reviewer_global_ids
                )
                for reviewer_local_id, reviewer_global_id in reviewer_ids:
                    username = (
                        reviewer_data[reviewer_global_id]['startUsername']
                    )
                    name = reviewer_data[reviewer_global_id]['names'][0]
                    name = f"{name} ({username})"
                    score = round(
                        optimization_problems[problem][submission_local_id]
                        [reviewer_local_id], 4
                    )
                    track_submission_info += [name, score]
                    global_submission_info += [username, score]
                    if args.area_chairs:
                        track_list = set(
                            reviewer_data[reviewer_global_id]['ac_tracks']
                        )
                    else:
                        track_list = set(
                            reviewer_data[reviewer_global_id]['tracks']
                        )
                    same_track = same_track and (track in track_list)

            # Similar reviewers
            reviewer_ids = zip(similar_reviewers, similar_reviewer_global_ids)
            for reviewer_local_id, reviewer_global_id in reviewer_ids:
                username = reviewer_data[reviewer_global_id]['startUsername']
                name = reviewer_data[reviewer_global_id]['names'][0]
                name = f"{name} ({username})"
                score = round(
                    optimization_problems[problem][submission_local_id]
                    [reviewer_local_id], 4
                )
                track_submission_info += [name, score]

            # Also append the track name to the global submission info
            global_submission_info += [track, same_track]

            # Get the COIs for the submission, and filter out all those that are
            # SACs or ACs
            submission_cois = cois[submission_id]
            coi_idx = [
                x for x in range(submission_cois.shape[0])
                if submission_cois[x] == 1
            ]
            coi_sacs = [
                reviewer_data[idx]['startUsername']
                for idx in coi_idx if reviewer_data[idx]['seniorAreaChair'] and
                track in reviewer_data[idx]['sac_tracks']
            ]
            coi_acs = [
                reviewer_data[idx]['startUsername']
                for idx in coi_idx if reviewer_data[idx]['areaChair'] and
                track in reviewer_data[idx]['ac_tracks']
            ]
            track_submission_info.append('; '.join(coi_sacs))
            track_submission_info.append('; '.join(coi_acs))

            # Append the submission data to both the
            track_assignments[track].append(track_submission_info)
            global_assignments.append(global_submission_info)

        # Open the file path given in the arguments as the global assignment
        # spreadsheet, writing each row
        with open(args.assignment_spreadsheet, 'w+') as f:
            writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(global_header_info)
            for entry in global_assignments:
                writer.writerow(entry)

        # For each track, create a file as a csv spreadsheet for all the track
        # submissions and their reviewer assignments
        file_base, file_extension = (
            os.path.splitext(args.assignment_spreadsheet)[:2]
        )
        for track in track_assignments.keys():
            alphanum_track = '-'.join(re.split(r'[\W,:]+', track))
            filename = f'{file_base}_{alphanum_track}{file_extension}'
            with open(filename, 'w+') as f:
                writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow(track_header_info)
                for entry in track_assignments[track]:
                    writer.writerow(entry)


if __name__ == "__main__":
    main()
