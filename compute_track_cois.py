"""
Computes COIs for papers in each track with the senior programme committee,
SACs and ACs.  This creates a spreadsheet ("all_output_file") with all the
COIs. This can be loaded into Excel and used to create a pivot-table to 
view for each track a matrix of paper x AC/SAC. 

This is useful for PCs, e.g:
 - to look for papers that have COIs with one/all SACs, or many ACs
 - debug those with way too many COIs
And for SACs 
 - to adjust the AC assignments (as there are more COIs than listed in softconf)

Run as follows for EMNLP 2020, following softconf_extract:

    python3  compute_track_cois.py \
        --submission_file $dir/submissions.jsonl \
        --reviewer_file $dir/reviewers.jsonl \
        --bid_file $dir/cois.npy \
        --output_file $dir/cois-for-sacs.csv \
        --all_output_file $dir/cois-for-ac-sacs.csv

We only used the "all_output_file".
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
import pandas

from suggest_utils import calc_reviewer_db_mapping, print_text_report, print_progress, calc_reviewer_id_mapping
from suggest_reviewers import create_embeddings, calc_similarity_matrix, calc_aggregate_reviewer_score, create_suggested_assignment

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="A line-by-line JSON file of submissions")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A json file of reviewer names and IDs that can review this time")
    parser.add_argument("--bid_file", type=str, default=None, help="A file containing numpy array of bids (0 = COI, >1 = no COI).")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path for CSV result sheet")
    parser.add_argument("--all_output_file", type=str, default=None, help="Output file path for CSV result sheet")

    args = parser.parse_args()

    assert args.all_output_file
    assert args.output_file
    assert args.submission_file
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
        sacs_by_track = defaultdict(list)
        for i, data in enumerate(reviewer_data):
            if 'name' in data:
                data['names'] = [data['name']]
                del data['name']
            if data['areaChair'] or data['seniorAreaChair']:
                if data['areaChair']:
                    acs_by_track[data['track']].append(i)
                if data['seniorAreaChair']:
                    sacs_by_track[data['track']].append(i)
                num_acs += 1
            num_ids = []
            # for id in data['ids']:
            #     try:
            #         num_ids.append(int(id))
            #     except ValueError:
            #         print(f"WARNING: failed to process id '{id}'' for {data['names']} username {data['startUsername']}")
            # data['ids'] = num_ids

        num_tracks = len(acs_by_track)
        print(f'WARNING: SAC and AC track mismatch') if set(sacs_by_track.keys()) != set(acs_by_track.keys()) else None
        print(f'WARNING: S/AC tracks and paper tracks mismatch') if set(sacs_by_track.keys()) != sub_tracks.union({'Multidisciplinary and AC COI'}) else None

    # Load and process COIs
    cois = np.where(np.load(args.bid_file) == 0, 1, 0) 

    rows_sac = []
    rows_all = []
    for i, query in enumerate(submissions):
        row = dict({'Submission ID': query['startSubmissionId'], 'Title': query['title'], 'Track': query['track']})            
        for track in sacs_by_track.keys():
            coi = []
            for sac in sacs_by_track[track]:
                rev = reviewer_data[sac]
                if cois[i,sac]:
                    coi.extend(rev['names'])
            row[f'COI-{track}'] = ';'.join(coi)
        rows_sac.append(row)

        for track in sorted(sacs_by_track.keys()):
            for sac in sacs_by_track[track]:
                rev = reviewer_data[sac]
                if cois[i,sac]:
                    rows_all.append({'Submission ID': query['startSubmissionId'], 'Paper Track': query['track'], 'Role':'SAC',
                                     'User Track': track, 'Username': rev['startUsername'], 'Name': ';'.join(rev['names'])})
            for ac in acs_by_track[track]:
                rev = reviewer_data[ac]
                if cois[i,ac]:
                    rows_all.append({'Submission ID': query['startSubmissionId'], 'Paper Track': query['track'], 'Role':'AC',
                                     'User Track': track, 'Username': rev['startUsername'], 'Name': ';'.join(rev['names'])})

    results = pandas.DataFrame(rows_sac)
    results.to_csv(args.output_file, quoting=2, index=False)

    results_all = pandas.DataFrame(rows_all)
    results_all.to_csv(args.all_output_file, quoting=2, index=False)

