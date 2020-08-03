import argparse
import json
import csv
import suggest_utils
import sys
import numpy as np
import re
import pandas
import os.path
import pickle

def find_colids(colnames, header):
    colids, colmap = [-1 for _ in colnames], {}
    for i, k in enumerate(colnames):
        for alias in k.split('|'):
            colmap[alias] = i
    for i, col in enumerate(header):
        if col in colmap:
            colids[colmap[col]] = i
    if any([x == -1 for x in colids]):
        print(f'WARNING: Couldn\'t find column ids in {colnames} {colids}', file=sys.stderr)
    return colids

def validate_track_name(tn):
    with open('tracks.txt','r') as infile:
        tracks=set([l.strip() for l in infile.readlines()])
        if tn in tracks:
            return True
    return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_in", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--profile_in", type=str, required=True, help="The profile CSV file")
    parser.add_argument("--bid_in", type=str, default=None, help="The submission numpy file")
    parser.add_argument("--reviewer_out", type=str, required=True, help="The track-specific reviewer json file")
    parser.add_argument("--submission_out", type=str, default=None, help="The track-specific submission json file")
    parser.add_argument('--submission_to_bid', default='submission_to_bid.txt', help='output file for submission id map to bid index')
    parser.add_argument("--bid_out", type=str, default=None, help="The track-specific submission numpy file")
    parser.add_argument("--tracks_file", type=str, default='tracks.txt', help="The list of tracks")
    parser.add_argument("--committee_list", type=str, help="ordered list of committee members--same order as bids columns (from COI detection stuff)")
    parser.add_argument("--track", type=str, help='current track')

    args = parser.parse_args()
    
    #NOTE HARD-CODED u_tracks file
    if os.path.exists(args.tracks_file):
        with open(args.tracks_file,'r') as tfile:
            with open('u_tracks.txt','r') as t2file:
                tracks=[x.strip() for x in tfile.readlines()]
                t2list=[x.strip() for x in t2file.readlines()]
                tdict=dict(zip(t2list,tracks))
    else:
        print('No tracks file.  Exiting...')

    # Get framework for bids
    if args.bid_in:
        bids=np.load(args.bid_in)
    else:
        print('Bids file missing.')
        sys.exit()
    
    if args.submission_to_bid:
        submission_to_bid = {}
        with open(args.submission_to_bid, 'r') as fin:
            for lines in fin:
                sid, bid_idx = lines.strip().split('\t')
                submission_to_bid[sid] = int(bid_idx)
                submission_to_bid[int(sid)] = int(bid_idx)
        print('load submission_to_bid.')
    else:
        print('submission_to_bid is missing.')
        sys.exit()
    # Process reviewers
    #Track managers (SAC) have, "Is a track manager" = "Yes"
    #ACs have, "Is a track manager" = "No", and "Roles"="committee:[trackname]"
    #Reviewers have "Is a track manager" = "No", and "Roles"="[trackname]"
    #Here we remove people/papers from tracks in the assignment process by putting a 0 in their bids column
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles', 'level', 'Is a track manager'] #NS added level
    ucol, ecol, fcol, lcol, scol, rcol, level_col, tman = find_colids(colnames, csv_lines[0])
    reviewers, reviewer_map, profile_map = [], {}, {}
    usernames=[]
    emails=[]
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]): #iterating through profiles
            s2id = line[scol].split('/')[-1]
            # Author data
            data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col]), 'Roles':line[rcol], 'Is a track manager':line[tman]}
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data
            
            if tdict[args.track] in line[rcol]:
                if 'manager' not in line[rcol]:
                    rev_data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col]), 'Roles':line[rcol], 'Is a track manager':line[tman]}
                    #print(json.dumps(rev_data), file=f)
                    reviewer_map[line[ucol]] = len(reviewers)
                    reviewer_map[line[ecol]] = len(reviewers)
                    reviewers.append(rev_data)
                    print(json.dumps(rev_data), file=f)

    # Process submissions (if present)
    if not args.submission_in:
        print('No submission file. Exiting...')
        sys.exit(0)
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    #colnames = ['Submission ID', 'Title', 'Abstract|Summary', 'Authors', 'All Author Emails', 'Track']
    #colnames = ['Submission ID', 'Title', 'Abstract', 'Authors', 'All Author Emails', 'Track']
    colnames = ['Submission ID', 'Title', 'Abstract', 'Authors', 'All Author Emails', 'Track', 'Acceptance Status']
    scol, tcol, abscol, acol, ecol, trcol, ascol = find_colids(colnames, csv_lines[0])
    submissions, submission_map = [], {}

    #load committee
    committee_list=pickle.load(open(args.committee_list,'rb'))

    # Write submissions
    missing = 0
    submission_num = 0
    bid_idx_list = []
    delim_re = re.compile(r'(?:, | and )')
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]): #order of submissions
            if line[trcol] != tdict[args.track]:
                continue
            if 'Reject' in line[ascol] or 'Reject' in line[trcol]:
                print('Reject in Acceptance Status')
                continue
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            authors = []
            for ae, an in zip(author_emails, author_names):
                if ae in profile_map:
                    authors.append(profile_map[ae])
                else:
                    authors.append({'name': an, 'ids': []})
                    missing+=1
                    
            data = {'title': line[tcol], 'paperAbstract': line[abscol], 'authors': authors, 'startSubmissionId': line[scol], 'Track':line[trcol]}
            bid_idx_list.append(submission_to_bid[line[scol]])
            submissions.append(data)
            submission_map[line[scol]] = i
            print(json.dumps(data), file=f)
            submission_num += 1
            
    #Track managers (SAC) have, "Is a track manager" = "Yes"
    #ACs have, "Is a track manager" = "No", and "Roles"="committee:[trackname]"
    #Reviewers have "Is a track manager" = "No", and "Roles"="[trackname]"
            num_reviewers=0
            num_acs=0
            num_sacs_managers=0
            #print(committee_list)
            for j in range(len(committee_list)):
                reviewer=committee_list[j]
                
                if reviewer in reviewer_map and 'manager' in reviewers[reviewer_map[reviewer]]['Roles']: #SAC, PC chair, GC
                    bids[i,j] = 0
                    num_sacs_managers+=1
        print('Print Submission for track: %s num: %d' % (tdict[args.track], submission_num))
        print('num_sacs_managers_num: %d' % num_sacs_managers)
        print('final i :%d, and all_bid_len: %d, bid_idx_list_len: %d' % (i, bids.shape[0], len(bid_idx_list)))

    print('Authors missing',missing)
    if args.bid_out:
        #bid_idx_list.sort()
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids[bid_idx_list])
    
