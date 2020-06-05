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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_in", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--profile_in", type=str, required=True, help="The profile CSV file")
    parser.add_argument("--bid_in", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--reviewer_out", type=str, required=True, help="The reviewer CSV file")
    parser.add_argument("--submission_out", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--bid_out", type=str, default=None, help="The submission CSV file")

    args = parser.parse_args()

    # Process reviewers
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles']
    ucol, ecol, fcol, lcol, scol, rcol = find_colids(colnames, csv_lines[0])
    # also check that they've agreed to review in the local profile questions
    Rcol, Ecol = find_colids(['PCRole', 'emergencyReviewer'], csv_lines[0])
    reviewers, reviewer_map, profile_map = [], {}, {}
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            s2id = line[scol].split('/')[-1] # FIXME: is there better parsing of these from the other repository?
            # Author data
            data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol]}
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data
            if len(line) < Ecol+1:
                line.extend([''] * (Ecol+1))

            is_area_chair = 'Meta Reviewer' in line[rcol] 
            is_senior_area_chair = '(manager' in line[rcol]
            is_programme_chair = 'committee:manager' in line[rcol]
            is_reviewer = 'committee:' in line[rcol] and not is_senior_area_chair and not is_programme_chair
            agreed_tscs = len(line) >= Ecol+1 and ('reviewer' in line[Rcol] or 'Yes' in line[Ecol])

            # check for not author, and not PC "manager:committee" and not SAC "committee:<track name> (manager #)"
            if is_reviewer and agreed_tscs or is_area_chair or is_senior_area_chair:
                track = re.sub(r'committee:', '', line[rcol])
                track = re.sub(r':?Meta Reviewer:?', '', track).strip()
                track = re.sub(r'^:', '', track).strip()
                if is_senior_area_chair:
                    # sigh, some track names contain the separator symbol ':'
                    track = re.sub(r'(.+):\1 \(manager.*', r'\1', track)
                rev_data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol],
                            'areaChair': is_area_chair, 'emergency': ('Yes' in line[Ecol]), 'track': track,
                            'seniorAreaChair': is_senior_area_chair}
                print(json.dumps(rev_data), file=f)
                reviewer_map[line[ucol]] = len(reviewers)
                reviewer_map[line[ecol]] = len(reviewers)
                reviewers.append(rev_data)

            # FIXME: experience / graduation year is also useful -- but maybe we will handle this separately

    # Process submissions 
    if not args.submission_in: sys.exit(0)
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Submission ID', 'Title', 'Track', 'Submission Type', 'Abstract|Summary', 'Authors', 'All Author Emails']
    scol, tcol, rcol, ycol, abscol, acol, ecol = find_colids(colnames, csv_lines[0])
    icols = find_colids([f'{i}: Username' for i in range(1, 99)], csv_lines[0])
    submissions, submission_map = [], {}

    # Load up SoftConf bids
    #   this can be straight from softconf "Bid_Information.csv"
    #   but better to use the reviewer-coi-detection code which adds additional COIs
    bids = np.full( (len(csv_lines)-1, len(reviewers)), 2 ) # Initialize with "Maybes"
    if args.bid_in:
        bids_in = pandas.read_csv(args.bid_in, skipinitialspace=True, index_col='Submission ID/Username')
        if bids_in.columns[-1].startswith("Unnamed:"):
            bids_in.drop(bids_in.columns[-1], axis=1, inplace=True)

        for i, (sid, sub_bids) in enumerate(bids_in.iterrows()):
            # process bids (even with no bidding, this still has COIs)
            for username, bidding_code in sub_bids.to_dict().items():
                if bidding_code in '1234':
                    reviewer_id = reviewer_map.get(username)
                    if reviewer_id != None:
                        bids[i, reviewer_id] = 4 - int(bidding_code)
                        # softconf has 1 = Yes, 2 = Maybe, 3 = No, 4 = COI
                        # this code base uses 0 = COI, 1 = No, 2 = Maybe, 3 = Yes
                    else:
                        # this reviewer is an SAC or similar
                        assert username in profile_map
        
    # Write submissions
    delim_re = re.compile(r'(?:, | and )')
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            #print(i,line)
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            author_startids = [line[icols[j]] for j in range(len(author_emails))]
            authors = []
            for ae, an, ai in zip(author_emails, author_names, author_startids):
                aep = profile_map.get(ae)
                aip = profile_map.get(ai)

                if aep or aip:
                    authors.append(aep or aip)
                    aer = reviewer_map.get(ae)
                    air = reviewer_map.get(ai)
                    if aer != None or air != None:
                        if aer != None:
                            bids[i,aer] = 0
                        else:
                            bids[i,air] = 0
                else:
                    print(f'WARNING: could not find account for {ae}, just using name {an}; username is "{ai}"')
                    authors.append({'name': an, 'ids': []})
    
            track = line[rcol]
            assert track

            if 'short' in line[ycol].lower():
                typ = 'short'
            elif 'long' in line[ycol].lower():
                typ = 'long'
            else:
                raise ValueError(f'Illegal Submission Type {line[ycol]}')

            data = {'title': line[tcol], 'track': track, 'type': typ, 'paperAbstract': line[abscol], 'authors': authors, 'startSubmissionId': line[scol]}
            submissions.append(data)
            submission_map[line[scol]] = i
            print(json.dumps(data), file=f)

    if args.bid_out:
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids)
