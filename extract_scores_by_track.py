"""
Outputs the similarity matrix for each paper x reviewer, to aid SACs with reviewer assignment.

Assumes the assignment code has been run and the big matrices have been
created, and code is run from the output folder. (sorry, was written in a
rush!) 
"""


import numpy as np
import json
from collections import *
import sys
from slugify import slugify
import pandas as pd

submission_file = 'submissions.jsonl'
with open(submission_file, "r") as f:
    submissions = [json.loads(x) for x in f]
    submission_abs = [x['paperAbstract'] for x in submissions]
    submission_index = dict([(x['startSubmissionId'], x) for x in submissions])

just_papers = set()
track_papers = defaultdict(list)
for i, submission in enumerate(submissions):
    if just_papers and submission['startSubmissionId'] not in just_papers:
        continue # skip over this paper     
    if submission['track']:
        track_papers[submission['track']].append(i)
    else:
        raise ValueError(f'Submission {submission["startSubmissionId"]} has no track assigned')

reviewer_file='reviewers.jsonl'
with open(reviewer_file, "r") as f:
    reviewer_data_orig = [json.loads(x) for x in f]
    reviewer_data = []
    reviewer_remapping = {}
    for i, data in enumerate(reviewer_data_orig):
        if not data['areaChair'] and not data['seniorAreaChair'] and not data['emergency']: 
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

track_reviewers = defaultdict(list)
for j, reviewer in enumerate(reviewer_data):
    track_reviewers[reviewer['track']].append(j)
tracks_for_assignment = sorted(track_papers.keys())

cois = np.where(np.load('cois.npy') == 0, 1, 0)
reviewer_scores = np.load('agg_matrix.npy')
if cois is not None:
    paper_rows, rev_cols = cois.nonzero()
    num_cois = 0
    for i in range(len(paper_rows)):
        pid, rid = paper_rows[i], rev_cols[i]
        acid = reviewer_remapping[rid]
        if acid != -1:
            reviewer_scores[pid,acid] = -1
            num_cois += 1
    print(f'Applying {num_cois} COIs', file=sys.stderr)


for track in tracks_for_assignment:
    ps = track_papers[track]
    rs = track_reviewers[track]
    if not ps or not rs: 
        print(f'WARNING skipping empty track {track}')
        continue
    trs = reviewer_scores[ps,:][:,rs]
    bests = np.argmax(trs,axis=1)
    names = [reviewer_data[rs[r]]['startUsername'] for r in bests]
    scores = trs[np.arange(trs.shape[0]),bests]
    data = np.hstack([np.array([names]).T, np.array([scores]).T, trs])
    df = pd.DataFrame(data, index=[submissions[p]['startSubmissionId'] for p in ps], columns=['best username', 'best score'] + [reviewer_data[r]['startUsername'] for r in rs])
    df.to_csv(f'scores-{slugify(track)}.csv')
