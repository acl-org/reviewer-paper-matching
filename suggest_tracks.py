import json
from model_utils import Example, unk_string
from sacremoses import MosesTokenizer
import argparse
from models import load_model
import numpy as np
import cvxpy as cp
import sys
from collections import defaultdict
import pandas

from suggest_utils import calc_reviewer_db_mapping, print_text_report, print_progress, calc_reviewer_id_mapping
from suggest_reviewers import create_embeddings, calc_similarity_matrix, calc_aggregate_reviewer_score, create_suggested_assignment

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line JSON file of submissions")
    parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A json file of reviewer names and IDs that can review this time")
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--filter_field", type=str, default="name", help="Which field to use as the reviewer ID (name/id)")
    parser.add_argument("--aggregator", type=str, default="weighted_top5", help="Aggregation type (max, weighted_topN where N is a number)")
    parser.add_argument("--bid_file", type=str, default=None, help="A file containing numpy array of bids (0 = COI, >1 = no COI).")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path for CSV result sheet")
    parser.add_argument("--save_paper_matrix", help="A filename for where to save the paper similarity matrix")
    parser.add_argument("--load_paper_matrix", help="A filename for where to load the cached paper similarity matrix")
    parser.add_argument("--save_aggregate_matrix", help="A filename for where to save the reviewer-paper aggregate matrix")
    parser.add_argument("--load_aggregate_matrix", help="A filename for where to load the cached reviewer-paper aggregate matrix")

    args = parser.parse_args()

    assert args.output_file
    assert args.submission_file
    assert args.db_file
    assert args.reviewer_file

    # Load the data
    sub_tracks = set()
    with open(args.submission_file, "r") as f:
        submissions = [json.loads(x) for x in f]
        submission_abs = [x['paperAbstract'] for x in submissions]
        for x in submissions:
            sub_tracks.add(x['track'])

    num_acs = 0
    with open(args.reviewer_file, "r") as f:
        reviewer_data = [json.loads(x) for x in f]
        acs_by_track = defaultdict(list)
        reviewers_by_track = defaultdict(list)
        sacs_by_track = defaultdict(list)
        for data in reviewer_data:
            if 'name' in data:
                data['names'] = [data['name']]
                del data['name']
            if data['areaChair'] or data['seniorAreaChair']:
                acs_by_track[data['track']].extend(data['names'])
                if data['seniorAreaChair']:
                    sacs_by_track[data['track']].extend(data['names'])
                num_acs += 1
            reviewers_by_track[data['track']].extend(data['names'])
        reviewer_names = [x['names'][0] for x in reviewer_data]

        num_tracks = len(acs_by_track)
        assert set(sacs_by_track.keys()) == set(acs_by_track.keys())
        #assert set(sacs_by_track.keys()) == sub_tracks # there's a COI track, with no papers (yet)
        
        # FIXME: someone has AC roles 'Information Extraction:NLP Applications'

    with open(args.db_file, "r") as f:
        db = [json.loads(x) for x in f]  # for debug
        db_abs = [x['paperAbstract'] for x in db]

    # create binary matrix of reviewer x paper
    rdb = calc_reviewer_db_mapping(reviewer_data, db, author_col=args.filter_field, author_field='authors')

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

    k = 10
    rows = []
    for i, query in enumerate(submissions):
        scores = reviewer_scores[i]
        row = dict({'Submission ID': query['startSubmissionId'], 'Title': query['title'], 'Track': query['track']})
        for track, reviewers in reviewers_by_track.items():
            # find best reviewers, as mean similarity of top k reviewers in track
            # FIXME: bug here, must of deleted some code for reviewer_id_map
            rids = [reviewer_id_map[id] for ids in reviewers for id in ids]
            rscores = scores[rids]
            tscore = np.mean(rscores.argsort()[-k:])
            row[track] = tscore
            
        if not cois is None:
            for track in tracks:
                coi = []
                for sac in sacs_by_track[track]:
                    ids = reviewer_id_map[sac]
                    if not np.any(cois[i,ids]):
                        coi = sac
                row[f'COI-{track}'] = ';'.join(coi)
        rows.append(row)

    results = pandas.DataFrame(rows)
    results.to_csv(args.output_file, quoting=2)
