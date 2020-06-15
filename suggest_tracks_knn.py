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
    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path for CSV result sheet")

    args = parser.parse_args()

    assert args.output_file
    assert args.submission_file
    assert args.model_file

    # Load the data
    submissions_by_track = defaultdict(list)
    print('Loading submissions', file=sys.stderr)
    with open(args.submission_file, "r") as f:
        submissions = [json.loads(x) for x in f]
        submission_abs = [x['paperAbstract'] for x in submissions]
        for i, x in enumerate(submissions):
            submissions_by_track[x['track']].append(i)
    tracks = list(sorted(submissions_by_track.keys()))
    print(f'{len(submissions)} papers and {len(tracks)} tracks', file=sys.stderr)
            
    print('Loading model', file=sys.stderr)
    model, epoch = load_model(None, args.model_file, force_cpu=True)
    model.eval()
    assert not model.training

    print('Embedding abstracts and computing paper x paper similarity', file=sys.stderr)
    embeddings = create_embeddings(model, submission_abs)
    similarities = np.matmul(embeddings, np.transpose(embeddings))
    np.fill_diagonal(similarities, 0)

    k = 15
    tracks_by_submission = np.zeros(len(submissions), dtype=np.int32)
    for i, x in enumerate(submissions):
        tracks_by_submission[i] = tracks.index(x['track'])

    print(f'Computing track assignments', file=sys.stderr)
            
    rows = []
    for i, x in enumerate(submissions):
        sims = similarities[i,:] 
        nids = np.argsort(sims)[-k:]
        ntracks = tracks_by_submission[nids]

        # find most common track
        (values, counts) = np.unique(ntracks,return_counts=True)
        ind = np.argmax(counts)
        track = tracks[values[ind]]

        row = dict(x)
        row['suggested track'] = track
        row['neighbour entropy'] = -np.sum(np.log(counts / k))
        rows.append(row)

    results = pandas.DataFrame(rows)
    results = results.reindex(columns=['startSubmissionId','title','paperAbstract','authors','type','track','suggested track','neighbour entropy'])
    results.to_csv(args.output_file, quoting=2)