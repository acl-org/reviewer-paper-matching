import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict

import cvxpy as cp
import numpy as np
import pandas as pd
from sacremoses import MosesTokenizer

from model_utils import Example, unk_string
from models import load_model
from suggest_utils import (
    calc_reviewer_db_mapping, print_progress, print_text_report
)

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')

# ------------------------------------------------------------------------------
# Function Definitions
# ------------------------------------------------------------------------------


def parse_args():
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
    return args


def create_embeddings(model, examps):
    """
    Embed textual examples

    Args:
        examps: A list of text to embed
    
    Returns:
        numpy.ndarray: A len(examps) by embedding size numpy matrix of
        embeddings
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

    Args:
        rdb (numpy.ndarray): Reviewer DB. Matrix of DB papers by reviewers
        scores (numpy.ndarray): Matrix of similarity scores between the current
            papers (rows) and the DB papers (columns)
        operator (str): Which operator to apply (max, weighted_topK)
    
    Returns:
        numpy.ndarray: Matrix of length reviewers indicating the score for that
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


def assign_quotas(
    reviewers,
    quota_file,
    max_papers,
    area_chairs=False,
):
    """
    Read in a quota file to create a mapping from the reviewer index to the max
    number of papers that reviewer has agreed to read

    Missing values are filled with the default max number of papers, and
    ``None`` is replaced with 0

    Args:
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
        quota_file (str): The name of the file containing the quota information
            for all reviewers. Must include fields ``Username`` and 
            ``QuotaForReview``
        max_papers (int): The default max number of papers to assign each
            reviewer
        area_chairs (bool): Whether or not this assignment is for ACs
    
    Returns:
        quotas (dict[int, int]): A mapping from each reviewer's global index to
        the max number of papers to be assigned to them
    """
    quotas = {}
    username_to_idx = dict(
        [(r['startUsername'], j) for j, r in enumerate(reviewers)]
    )
    quota_table = pd.read_csv(
        quota_file, skipinitialspace=True, quotechar='"', encoding="UTF-8"
    )

    # Quota file will usually be a csv downloaded from the Author/Reviewer
    # Profiles section of the softconf spreadsheet maker. Thus it will include
    # both authors and reviewers, and only the reviewers are relevant. Here,
    # filter only the reviewers, fill any nan values with the default reviewer
    # quota, and replace the string value 'None' with 0
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
    Augment the reviewer-paper aggregate similarity matrix with penalties for
    regular reviewers if assigning only to ACs, or for (S)ACs if assigning to 
    regular reviewers. Count the number of reviewers included and excluded

    Note, these (soft) penalties are mostly redundant since the wrong types of
    reviewers are also excluded when forming the matrices for the optimization
    problem(s) (see ``split_by_subproblem``), but applying the penalties is
    useful for transparency if the wrong type of reviewer "slips through"

    Args:
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
        reviewer_scores (numpy.ndarray): The submission x reviewer matrix
            containing aggregate similarity scores
        area_chairs (bool): Whether or not this assignment is for ACs

    Returns:
        reviewer_scores (numpy.ndarray): The submission x reviewer matrix
            augmented with penalties to exclude ACs or regular reviewers as
            necessary
        num_included (int): The number of reviewers included (i.e. not 
            penalized)
        num_excluded (int): The number of reviewers excluded (i.e. penalized)
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


def get_track_sacs(reviewers):
    """
    Get the list of SACs for each track

    Args:
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
    
    Returns:
        track_sacs (defaultdict[str, list]): A mapping from each track name to
        the list of SACs for that track
    """
    track_sacs = defaultdict(set)
    for reviewer in reviewers:
        sac_tracks = reviewer['sac_tracks']
        if len(sac_tracks) > 0:
            for track in sac_tracks:
                track_sacs[track].add(reviewer['startUsername'])
    return track_sacs


def split_by_subproblem(
    reviewers,
    submissions,
    reviewer_scores,
    quotas=None,
    by_track=False,
    area_chairs=False
):
    """
    Split the submission-reviewer matrix into subproblems for optimization

    If reviewers should only be assigned to reviewers within the same track,
    optimization runtime is *greatly* improved by splitting into separate
    optimizations for each track, rather than relying on soft constraints to
    rule out reviewers from other tracks. Otherwise, the matrix is broken down
    into a single optimization problem with the label ``all_tracks``

    Args:
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
        submissions (list[dict]): The list of jsonl-style dictionaries
            containing the submission data
        reviewer_scores (numpy.ndarray): The submission x reviewer matrix
            containing aggregate similarity scores
        quotas (dict[int, int]): A mapping from each reviewer's global index
            to the max number of papers to be assigned to them
        by_track (bool): Whether papers should only be assigned to reviewers
            within the same track. If ``True``, the optimization will be broken
            down into a separate problem for each track
        area_chairs (bool): Whether or not this assignment is for ACs

    Returns:
        optimization_problems (dict[str, numpy.ndarray]): A mapping from each
            track name to the reduced similarity matrix only including papers
            and reviewers from that track
        problem_papers (defaultdict[str, list]): A mapping from each track name
            to the list of its included papers (by indices)
        problem_reviewers (defaultdict[str, list]): A mapping from each track
            name to the list of its included reviewers (by indices)
        problem quotas (defaultdict[str, dict]): A mapping from each track name
            to a dictionary of the quotas for that track (inner dictionary
            hashed by the *local index of the reviewer)
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

    Args:
    reviewer_scores (numpy.ndarray): The similarity scores used to assign
        reviewers
    reviews_per_paper (int): Maximum number of reviews per paper
    min_papers_per_reviewer (int): Minimum number of reviews per paper
    max_papers_per_reviewer (int): Maximum number of papers per reviewer
    quotas (dict): Per-reviewer quota on the maximum number of papers
    anonymity_multiplier: A multiplier in how many extra papers to assign
        (for anonymization purposes)

    Returns:
        An assignment of papers to reviewers, and a score indicating the
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


def parse_assignments(
    submissions,
    paper_similarity_matrix,
    optimization_matrices,
    problem_papers,
    problem_reviewers,
    problem_assignments,
    by_track=False,
    num_assigned=3,
    num_similar=3
):
    """
    For each paper, use the optimization assignment matrices and other data
    objects to parse the reviewer assignment to an easy-to-call dictionary of
    the assignment information (i.e. assigned reviewers, similar reviewers,
    scores, track)

    Args:
        submissions (list[dict]): The list of jsonl-style dictionaries
            containing the submission data
        paper_similarity_matrix (numpy.ndarray): A matrix of similarity scores
            between the submission papers (rows) and the papers in the ACL
            database (columns)
        optimization_matrices (defaultdict[str, np.ndarray]): A mapping from
            each track name to a similarity matrix of papers and reviewers from
            that track
        problem_papers (defaultdict[str, list]): A mapping from each track name
            to the list of its included papers (by indices)
        problem_reviewers (defaultdict[str, list]): A mapping from each track
            name to the list of its included reviewers (by indices)
        problem_assignments (defaultdict[str, list]): A mapping from each track
            name to the resulting matrix of its optimization problem (from
            which the assignment is read)
        by_track (bool): Whether papers are only assigned to reviewers
            within the same track. If ``True``, the optimization problem has
            been broken into multiple subproblems, each with its own matrix
        num_assigned (int): The number of reviewers/ACs to assign based on the
            optimization
        num_similar (int): The number of similar reviewers/ACs to list for each
            submission in addition to the assigned reviewers

    Returns:
        list[dict]: A list of dictionaries, one per submission, containing all
        the labeled fields required for querying and outputing the assignments
    """
    global_assignments = []
    problem = 'all_tracks'
    # Loop over the submissions
    for submission_idx, submission in enumerate(submissions):

        # Get the track and submission id for each
        track = submission['track']
        if by_track:
            problem = track
        submission_local_idx = problem_papers[problem].index(submission_idx)
        start_idx = submission['startSubmissionId']

        # Use the scores to get the index for each assigned reviewer, and
        # get their username for the output spreadsheet
        try:
            assigned_reviewer_local_ids = (
                problem_assignments[problem][submission_local_idx].argsort()
                [-num_assigned:][::-1]
            )
        except:
            assigned_reviewer_local_ids = []
        assigned_reviewer_global_ids = [
            problem_reviewers[problem][j] for j in assigned_reviewer_local_ids
        ]
        assigned_reviewer_scores = [
            optimization_matrices[problem][submission_local_idx][reviewer_idx]
            for reviewer_idx in assigned_reviewer_local_ids
        ]

        similar_reviewer_local_ids = (
            optimization_matrices[problem][submission_local_idx].argsort()
            [-num_similar:][::-1]
        )
        similar_reviewer_global_ids = [
            problem_reviewers[problem][j] for j in similar_reviewer_local_ids
        ]
        similar_reviewer_scores = [
            optimization_matrices[problem][submission_local_idx][reviewer_idx]
            for reviewer_idx in similar_reviewer_local_ids
        ]

        # Get most similar papers
        paper_sim_scores = paper_similarity_matrix[submission_idx]
        similar_paper_global_ids = (
            paper_sim_scores.argsort()[-num_similar:][::-1]
        )
        similar_paper_scores = [
            paper_sim_scores[idx] for idx in similar_paper_global_ids
        ]

        assignment_data = {
            'start_idx': start_idx,
            'track': track,
            'similar_paper_global_ids': similar_paper_global_ids,
            'similar_paper_scores': similar_paper_scores,
            'assigned_reviewer_global_ids': assigned_reviewer_global_ids,
            'assigned_reviewer_scores': assigned_reviewer_scores,
            'similar_reviewer_global_ids': similar_reviewer_global_ids,
            'similar_reviewer_scores': similar_reviewer_scores
        }

        global_assignments.append(assignment_data)

    return global_assignments


def get_jsonl_rows(assignments, submissions, reviewers, db_papers):
    """
    Get the list of data to write to the output jsonl file

    Args:
        assignments (list[dict]): The per-submission assignment data
        submissions (list[dict]): The list of jsonl-style dictionaries
            containing the submission data
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
        db_papers: The list of jsonl-style dictionary containing the information
            on the ACL-Anthology papers
        
    Returns:
        list[dict]: The list of jsonl-style rows to be written, containing the
            reviewer assignment
    """

    data = []

    for submission_idx, submission in enumerate(assignments):

        ret_dict = dict(submissions[submission_idx])

        # Add information on the top similar papers
        best_paper_info = zip(
            submission['similar_paper_global_ids'],
            submission['similar_paper_scores']
        )
        ret_dict['similarPapers'] = [
            {
                'title': db_papers[idx]['title'],
                'paperAbstract': db_papers[idx]['paperAbstract'],
                'score': score
            } for idx, score in best_paper_info
        ]

        # Add information on the top similar reviewers
        ret_dict['topSimReviewers'] = []
        similar_reviewer_info = zip(
            submission['similar_reviewer_global_ids'],
            submission['similar_reviewer_scores']
        )
        for idx, score in similar_reviewer_info:
            next_dict = dict(reviewers[idx])
            next_dict['score'] = score
            ret_dict['topSimReviewers'].append(next_dict)

        # Add information on the assigned reviewers
        ret_dict['assignedReviewers'] = []
        assigned_reviewer_info = zip(
            submission['assigned_reviewer_global_ids'],
            submission['assigned_reviewer_scores']
        )
        for idx, score in assigned_reviewer_info:
            next_dict = dict(reviewers[idx])
            next_dict['score'] = score
            ret_dict['assignedReviewers'].append(next_dict)

        data.append(ret_dict)

    return data


def get_csv_header(
    reviews_per_paper, num_similar, area_chairs=False, is_global=False
):
    """
    Get the csv output header fields, based on the number of assigned and
    similar reviewers

    Args:
        reviews_per_paper (int): Number of reviews received by each paper
        num_similar (int): The number of similar reviewers/ACs to list for each
            submission in addition to the assigned reviewers
        area_chairs (bool): Whether or not this assignment is for ACs
        is_global (bool): whether or not this header is for the global csv file
            (if ``False``, assumed to be for the track-wise file)

    Returns:
        header (list): Header fields for the output csv file
    """

    if area_chairs:
        role = 'AC'
    else:
        role = 'reviewer'

    # Include ID and assigned reviewers in both the global and track-wise csv
    # headers
    header = ['ID']
    for x in range(1, reviews_per_paper + 1):
        reviewer_field = f'Assigned {role} {x}'
        score_field = f'Assigned {role} {x} score'
        header += [reviewer_field, score_field]

    # To the global header, add track and a flag for whether all reviewers are
    # in the same track. To the track-wise header, add non-assigned similar
    # reviewers, as well as SACs and ACs with COIs
    if is_global:
        header += ['Track', 'Assigned within same track']
    else:
        for x in range(1, num_similar + 1):
            reviewer_field = f'Similar {role} {x}'
            score_field = f'Similar {role} {x} score'
            header += [reviewer_field, score_field]
        header += ['SACs with COI', 'ACs with COI']

    return header


def get_csv_rows(
    assignments, reviewers, cois, reviews_per_paper, area_chairs=False
):
    """
    Get the lists of data to write to the output csv files (global and 
    track-wise)

    Args:
        assignments (list[dict]): The per-submission assignment data
        reviewers (list[dict]): The list of jsonl-style dictionaries containing
            the reviewer data
        cois (numpy.ndarray): The matrix encoding the COI status between
            submissions (rows) and reviewers (columns)
        reviews_per_paper (int): Number of reviews received by each paper
        area_chairs (bool): Whether or not this assignment is for ACs
    
    Returns:
        global_rows (list[list]): The list of csv data rows to write to the
            global output spreadsheet
        global_softconf_uploadable (list[str]): The list of data strings to be
            written to the global softconf-uploadable output file
        track_rows (defaultdict[str, list[list]]): The mapping from track name
            to the list of csv data rows to write to the corresponding
            track-wise output spreadsheet
        track_softconf_uploadables (defaultdict[str, list[str]]): The mapping
            from track name to the list of data string to be written to the
            corresponding track-wise softconf-uploadable output file
        
    """
    global_rows = []
    global_softconf_uploadable = []
    track_rows = defaultdict(list)
    track_softconf_uploadables = defaultdict(list)

    track_sacs = get_track_sacs(reviewers)

    # Loop over the submissions
    for global_submission_id, submission in enumerate(assignments):

        # Get the track and submission id for each
        track_info = []
        global_info = []
        softconf_upload_string = []
        track = submission['track']
        start_idx = submission['start_idx']
        track_info.append(start_idx)
        global_info.append(start_idx)
        softconf_upload_string.append(start_idx)

        # Add assigned reviewer information to spreadsheet row
        assigned_reviewers = submission['assigned_reviewer_global_ids']
        same_track = True
        if len(assigned_reviewers) == 0:
            for i in range(reviews_per_paper):
                track_info += ['', '']
                global_info += ['', '']
                softconf_upload_string += ['']
        else:
            reviewer_info = zip(
                assigned_reviewers, submission['assigned_reviewer_scores']
            )
            for reviewer_idx, reviewer_score in reviewer_info:
                username = reviewers[reviewer_idx]['startUsername']
                name = reviewers[reviewer_idx]['names'][0]
                name = f"{name} ({username})"
                score = round(reviewer_score, 4)
                track_info += [name, score]
                global_info += [username, score]
                softconf_upload_string += [username]
                if area_chairs:
                    track_list = set(reviewers[reviewer_idx]['ac_tracks'])
                else:
                    track_list = set(reviewers[reviewer_idx]['tracks'])
                same_track = same_track and (track in track_list)

            # Add similar reviewer information to spreadsheet row
            reviewer_info = zip(
                submission['similar_reviewer_global_ids'],
                submission['similar_reviewer_scores']
            )
            for reviewer_idx, reviewer_score in reviewer_info:
                username = reviewers[reviewer_idx]['startUsername']
                name = reviewers[reviewer_idx]['names'][0]
                name = f"{name} ({username})"
                score = round(reviewer_score, 4)
                track_info += [name, score]

        # Also append the track name to the global submission info
        global_info += [track, same_track]

        # Get the COIs for the submission, and filter out all those that are
        # SACs or ACs
        submission_cois = cois[global_submission_id]
        coi_idx = [
            x
            for x in range(submission_cois.shape[0]) if submission_cois[x] == 1
        ]
        coi_sacs = [
            reviewers[idx]['startUsername']
            for idx in coi_idx if reviewers[idx]['seniorAreaChair'] and
            track in reviewers[idx]['sac_tracks']
        ]
        coi_acs = [
            reviewers[idx]['startUsername'] for idx in coi_idx if
            reviewers[idx]['areaChair'] and track in reviewers[idx]['ac_tracks']
        ]
        track_info.append('; '.join(coi_sacs))
        track_info.append('; '.join(coi_acs))

        # If all track SACs have a COI with the submission, move it to the
        # separate COI track
        if set(coi_sacs) == track_sacs[track]:
            track_info.append(track)
            track = 'COI'
            global_info[-2] = track

        # Add the formed rows to the respective spreadsheets
        global_rows.append(global_info)
        track_rows[track].append(track_info)

        softconf_upload_string = ':'.join(softconf_upload_string)
        global_softconf_uploadable.append(softconf_upload_string)
        track_softconf_uploadables[track].append(softconf_upload_string)

    return (
        (global_rows, global_softconf_uploadable),
        (track_rows, track_softconf_uploadables)
    )

# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------

def main():

    # --------------------------------------------------------------------------
    # Part 1: Read in the arguments
    # --------------------------------------------------------------------------

    args = parse_args()

    # --------------------------------------------------------------------------
    # Part 2: Load the data and calculate similarity between submissions and
    # reviewers
    # --------------------------------------------------------------------------

    data_load_start_time = time.time()

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

    data_load_end_time = time.time()
    data_load_time = round((data_load_end_time - data_load_start_time) / 60, 2)
    print(
        f"Time loading and preprocessing data: {data_load_time} minutes",
        file=sys.stderr
    )
    similarity_matrix_start_time = time.time()

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

    similarity_matrix_end_time = time.time()
    similarity_matrix_time = round(
        (similarity_matrix_end_time - similarity_matrix_start_time) / 60, 2
    )
    print(
        "Time calculating paper similarity matrix:"
        f" {similarity_matrix_time} minutes",
        file=sys.stderr
    )
    aggregation_start_time = time.time()

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

    aggregation_end_time = time.time()
    aggregation_time = round(
        (aggregation_end_time - aggregation_start_time) / 60, 2
    )
    print(
        "Time calculating aggregated similarity matrix:"
        f" {aggregation_time} minutes",
        file=sys.stderr
    )
    formulization_start_time = time.time()

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

    formulization_end_time = time.time()
    formulization_time = round(
        (formulization_end_time - formulization_start_time) / 60, 2
    )
    print(
        f"Time formulating optimization problem: {formulization_time} minutes",
        file=sys.stderr
    )
    optimization_start_time = time.time()

    # --------------------------------------------------------------------------
    # Part 5: Calculate a reviewer assignment based on the constraints
    # --------------------------------------------------------------------------

    problem_assignments = {}
    problem_scores = {}
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

        problem_assignments[problem] = assignment
        problem_scores[problem] = assignment_score

        print(
            f"Done calculating assignment. Total score: {assignment_score}",
            file=sys.stderr
        )
        if assignment is None:
            warnings.warn(
                f"No solution found for category {problem}", RuntimeWarning
            )

    optimization_end_time = time.time()
    optimization_time = round(
        (optimization_end_time - optimization_start_time) / 60, 2
    )
    print(
        "Time calculating optimal assignment of papers:"
        f" {optimization_time} minutes",
        file=sys.stderr
    )

    # --------------------------------------------------------------------------
    # Part 6: Parse the assignments into a dictionary of reviewer IDs and other
    # info for each submission
    # --------------------------------------------------------------------------

    global_assignments = parse_assignments(
        submissions=submissions,
        paper_similarity_matrix=mat,
        optimization_matrices=optimization_problems,
        problem_papers=problem_papers,
        problem_reviewers=problem_reviewers,
        problem_assignments=problem_assignments,
        by_track=args.track,
        num_assigned=args.reviews_per_paper,
        num_similar=args.num_similar_to_list
    )

    # --------------------------------------------------------------------------
    # Part 7: Print out the results in jsonl format
    # --------------------------------------------------------------------------

    jsonl_data = get_jsonl_rows(
        assignments=global_assignments,
        submissions=submissions,
        reviewers=reviewer_data,
        db_papers=db
    )

    with open(args.suggestion_file, 'w') as outf:
        for entry in jsonl_data:
            if args.output_type == 'json':
                print(json.dumps(entry), file=outf)
            elif args.output_type == 'text':
                print_text_report(entry, file=outf)
            else:
                raise ValueError(f'Illegal output_type {args.output_type}')

    print(
        f"Done creating suggestions, written to {args.suggestion_file}\n",
        file=sys.stderr
    )

    # --------------------------------------------------------------------------
    # Part 8 (optional): Print out the results in more human-readable
    # spreadsheets
    # --------------------------------------------------------------------------

    # ACL-2021: We are outputting an alternative data file to easily create
    # a per-track spreadsheet of assigned reviewers, as well as a global file
    # with the minimum assignment information
    if args.assignment_spreadsheet:

        global_header_info = get_csv_header(
            reviews_per_paper=args.reviews_per_paper,
            num_similar=args.num_similar_to_list,
            area_chairs=args.area_chairs,
            is_global=True
        )
        track_header_info = get_csv_header(
            reviews_per_paper=args.reviews_per_paper,
            num_similar=args.num_similar_to_list,
            area_chairs=args.area_chairs,
            is_global=False
        )
        coi_header_info = track_header_info + ['Original track']

        global_data, track_data = get_csv_rows(
            assignments=global_assignments,
            reviewers=reviewer_data,
            cois=cois,
            reviews_per_paper=args.reviews_per_paper,
            area_chairs=args.area_chairs
        )

        global_rows, global_softconf_uploadable = global_data
        track_rows, track_softconf_uploadables = track_data

        # Separate the input file base from its extension so we can print
        # multiple files with the same general schema
        file_base, file_extension = (
            os.path.splitext(args.assignment_spreadsheet)[:2]
        )

        # Open the file path given in the arguments as the global assignment
        # spreadsheet, writing each row
        with open(args.assignment_spreadsheet, 'w+') as f:
            writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(global_header_info)
            for entry in global_rows:
                writer.writerow(entry)
        with open(file_base + '.txt', 'w+') as f:
            for line in global_softconf_uploadable:
                print(line, file=f)

        # For each track, create a file as a csv spreadsheet for all the track
        # submissions and their reviewer assignments
        for track in track_rows.keys():
            alphanum_track = '-'.join(re.split(r'[\W,:]+', track))
            filename = f'{file_base}_{alphanum_track}{file_extension}'
            with open(filename, 'w+') as f:
                writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                if track == 'COI':
                    writer.writerow(coi_header_info)
                else:
                    writer.writerow(track_header_info)
                for entry in track_rows[track]:
                    writer.writerow(entry)
            filename = f'{file_base}_{alphanum_track}.txt'
            with open(filename, 'w+') as f:
                for line in track_softconf_uploadables[track]:
                    print(line, file=f)


if __name__ == "__main__":
    main()
