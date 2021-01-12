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


def is_area_chair_from(role):
    """ Computes if the role description corresponds to a area chair

    Usually, ACs have 'Meta Reviewer" in their role description, e.g. "committee:Summarization:Meta Reviewer".
    """
    return 'Meta Reviewer' in role


def is_senior_area_chair_from(role):
    """ Computes if the role description corresponds to a senior area chair

    SACs are discinguishable from everyone else via the "manager" suffix.
    E.g. committee:Speech:Speech (manager 1)
    """
    return '(manager' in role


def is_reviewer_from(role):
    """ Computes if the role description corresponds to a reviewer

    Before reviewer track assignment each reviewer has the role 'committee'
    and after - most of the reviewers have new format: 'committee:TRACK'
    where TRACK is the track name (note that track names may contain ':'-symbol).

    Note that if you do not want to assign papers to the reviewers without track,
    you can provide --track argument to the suggest_reviewers.py

    DO NOT MODIFY THIS FUNCTION UNLESS YOU FULLY UNDERSTAND THIS REPOSITORY.
    """
    if role == 'committee':
        return True
    
    if role.startswith('committee:') and not is_senior_area_chair_from(role) and not is_area_chair_from(role):
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

    args = parser.parse_args()

    # Process reviewers
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        profiles_csv = list(csv_reader)

    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles']
    ucol, ecol, fcol, lcol, scol, rcol = find_colids(colnames, profiles_csv[0])
    # also check that they've agreed to review in the local profile questions 
    Rcol, Ecol = find_colids(['PCRole', 'emergencyReviewer'], profiles_csv[0])
    reviewers, reviewer_map, profile_map = [], {}, {}
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(profiles_csv[1:]):
            s2id = line[scol].split('/')[-1] # FIXME: is there better parsing of these from the other repository?
            # Author data
            data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol]}
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data
            if len(line) < Ecol+1:
                line.extend([''] * (Ecol+1)) # some lines are incomplete (due to softconf change to profile form)

            _role = line[rcol]

            is_area_chair = is_area_chair_from(_role)
            is_senior_area_chair = is_senior_area_chair_from(_role)
            is_programme_chair = _role == "manager:committee"
            is_reviewer = is_reviewer_from(_role)

            agreed_tscs = 'reviewer' in line[Rcol] or 'AC' in line[Rcol] or 'Yes' in line[Ecol]
            if is_reviewer and not agreed_tscs:
                print(f'WARNING: {line[ucol]} has not agreed to review or emergency review; role {_role}; agreed {line[Rcol]}')
                # this line below does not make sense
                agreed_tscs = True

            # check for not author, and not PC "manager:committee" and not SAC "committee:<track name> (manager #)"
            if is_reviewer and agreed_tscs or is_area_chair or is_senior_area_chair:
                track = re.sub(r'committee:', '', _role)
                track = re.sub(r':?Meta Reviewer:?', '', track).strip()
                track = re.sub(r'^:', '', track).strip()

                if is_senior_area_chair:
                    # sigh, some track names contain the separator symbol ':'
                    track = re.sub(r'(.+):\1 \(manager.*', r'\1', track)

                rev_data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol],
                            'areaChair': is_area_chair, 'emergency': ('Yes' in line[Ecol] and 'no' == line[Rcol]), 
                            'track': track, 'seniorAreaChair': is_senior_area_chair}

                print(json.dumps(rev_data), file=f)
                reviewer_map[line[ucol]] = len(reviewers)
                reviewer_map[line[ecol]] = len(reviewers)
                reviewers.append(rev_data)

            # FIXME: experience / graduation year is also useful -- but maybe we will handle this separately

    # Process submissions 
    if not args.submission_in:
        sys.exit(0)

    with open(args.submission_in, 'r') as f:  # Submissions.csv
        csv_reader = csv.reader(f, delimiter=',')
        submissions_csv = list(csv_reader)

    colnames = ['Submission ID', 'Title', 'Track', 'Submission Type', 'Abstract', 'Authors', 'All Author Emails', 'Acceptance Status']
    scol, tcol, rcol, ycol, abscol, acol, ecol, stcol = find_colids(colnames, submissions_csv[0])
    icols = find_colids([f'{i}: Username' for i in range(1, 99)], submissions_csv[0])
    submissions = []
    submissions, submission_map = [], {}

    # Load up SoftConf bids
    #   this can be straight from softconf "Bid_Information.csv"
    #   but better to use the reviewer-coi-detection code which adds additional COIs
    bids = np.full( (len(submissions_csv)-1, len(reviewers)), 2 ) # Initialize with "Maybes"
    if args.bid_in:
        bids_in = pandas.read_csv(args.bid_in, skipinitialspace=True, index_col='Submission ID/Username')
        if bids.shape[0] != bids_in.shape[0]:
            raise RuntimeError("--bids_in should have the rows corresponding to the rows in the --submission_in")

        if bids_in.columns[-1].startswith("Unnamed:"):
            bids_in.drop(bids_in.columns[-1], axis=1, inplace=True)

        for i, (sid, sub_bids) in enumerate(bids_in.iterrows()):
            # process bids (even with no bidding, this still has COIs)
            for username, bidding_code in sub_bids.to_dict().items():

                if isinstance(bidding_code, float) and bidding_code != bidding_code:
                    print(f"NaN value found for username {username}")
                    continue

                if bidding_code in '1234':
                    reviewer_id = reviewer_map.get(username)
                    if reviewer_id != None:
                        bids[i, reviewer_id] = 4 - int(bidding_code)
                        # softconf has 1 = Yes, 2 = Maybe, 3 = No, 4 = COI
                        # this code base uses 0 = COI, 1 = No, 2 = Maybe, 3 = Yes

    # Write submissions
    delim_re = re.compile(r'(?:, | and )')
    not_found = set()
    submission_kept = []
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(submissions_csv[1:]):
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
                    if ai: not_found.add(ai)
            track = line[rcol]

            # assert track
            if not track:
                print(f'WARNING: did not find track for the author {author_names}')

            if 'short' in line[ycol].lower():
                typ = 'short'
            elif 'long' in line[ycol].lower():
                typ = 'long'
            else:
                raise ValueError(f'Illegal Submission Type {line[ycol]}')

            if 'Reject' not in line[stcol]:
                data = {'title': line[tcol], 'track': track, 'type': typ, 'paperAbstract': line[abscol], 
                        'authors': authors, 'startSubmissionId': line[scol]}
                submissions.append(data)
                #submission_map[line[scol]] = i
                print(json.dumps(data), file=f)
                submission_kept.append(i)

    if args.bid_out:
        # remove the bids for rejected submissions
        bids = bids[submission_kept, :]
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids)
