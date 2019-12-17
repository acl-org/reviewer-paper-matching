import argparse
import json
import csv
import suggest_utils
import sys
import numpy as np
import re
import pandas

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
    parser.add_argument("--bid_in", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--reviewer_out", type=str, required=True, help="The reviewer CSV file")
    parser.add_argument("--submission_out", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--bid_out", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--track_name", type=str, default=None, help="The track for assignment")
    parser.add_argument("--ac_assignment", type=bool, default=False, help="Assignment of ACs only?")

    args = parser.parse_args()
    
    if not args.track:
        print('There is no track chosen.  Doing assignment across all tracks.')
    elif validate_track_name(args.track_name):
        print(args.track_name, 'is not a valid track name.')
        sys.exit()

    # Process reviewers
    #Track managers (SAC) have, "Is a track manager" = "Yes"
    #ACs have, "Is a track manager" = "No", and "Roles"="committee:[trackname]"
    #Reviewers have "Is a track manager" = "No", and "Roles"="[trackname]"
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles', 'level', 'Is a track manager'] #NS added level
    ucol, ecol, fcol, lcol, scol, rcol, level_col = find_colids(colnames, csv_lines[0])
    reviewers, reviewer_map, profile_map = [], {}, {}
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            s2id = line[scol].split('/')[-1]
            # Author data
            data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col])}
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data
            
            #if 'committee' in line[rcol]:
            if line[rcole]!='Author':
                rev_data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col])}
                print(json.dumps(rev_data), file=f)
                reviewer_map[line[ucol]] = len(reviewers)
                reviewer_map[line[ecol]] = len(reviewers)
                reviewers.append(rev_data)

    # Process submissions (if present)
    if not args.submission_in:
        print('No submission file. Exiting...')
        sys.exit(0)
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Submission ID', 'Title', 'Abstract|Summary', 'Authors', 'All Author Emails']
    scol, tcol, abscol, acol, ecol = find_colids(colnames, csv_lines[0])
    submissions, submission_map = [], {}

    # Get framework for bids
#    bids = np.full( (len(csv_lines)-1, len(reviewers)), 2 ) # Initialize with "Maybes"
#    if args.bid_in:
#        bids=np.load('coibids.npy')
#    else:
#        print('Bids file missing.')

    # Write submissions
    delim_re = re.compile(r'(?:, | and )')
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            authors = []
            for ae, an in zip(author_emails, author_names):
                if ae in profile_map:
                    authors.append(profile_map[ae])
#                    if ae in reviewer_map:
#                        bids[i,reviewer_map[ae]] = 0
                else:
                    print(f'WARNING: could not find account for {ae}, just using name {an}')
                    authors.append({'name': an, 'ids': []})
            data = {'title': line[tcol], 'paperAbstract': line[abscol], 'authors': authors, 'startSubmissionId': line[scol]}
            submissions.append(data)
            submission_map[line[scol]] = i
            print(json.dumps(data), file=f)

#    if args.bid_out:
#        with open(args.bid_out, 'wb') as f:
#            np.save(f, bids)
