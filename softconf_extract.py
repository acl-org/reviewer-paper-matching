import argparse
import json
import csv
import suggest_utils
import sys
import numpy as np
import re
import pandas

def find_colids(colnames, header):
    '''Get the csv column index of each field in the colnames parameter'''
    colids, colmap = [-1 for _ in colnames], {}
    for i, name in enumerate(colnames):
        for alias in name.split('|'):
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

    csv.field_size_limit(sys.maxsize)

    # ----------------------------------------------------------------------------------------------
    # Process reviewer/author profile csv file from softconf and clean for output file
    # ----------------------------------------------------------------------------------------------

    # Read-in reviewer and author profiles and process field IDs
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles']
    ucol, ecol, fcol, lcol, scol, rcol = find_colids(colnames, csv_lines[0])
    # Also check that they've agreed to review in the local profile questions 
    Rcol, Ecol = find_colids(['PCRole', 'emergencyReviewer'], csv_lines[0])
    last_field = max([ucol, ecol, fcol, lcol, scol, rcol, Rcol, Ecol])
    reviewers, reviewer_map, profile_map = [], {}, {}

    # Loop through reviewer and author profiles
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            # Some lines are incomplete due to the rightward columns being blank
            if len(line) < last_field + 1:
                line.extend([''] * (last_field + 1 - len(line)))
            
            # FIXME: is there better parsing of these from the other repository?
            s2id = line[scol].split('/')[-1]
            
            # Map author/reviewer username and email to basic profile info
            data = {
                'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol]
            }
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data

            # Check for special reviewer roles, and whether they have agreed to emergency review
            is_area_chair = 'Meta Reviewer' in line[rcol] 
            is_senior_area_chair = '(manager' in line[rcol]
            is_programme_chair = line[rcol] == 'manager'
            #is_reviewer = 'committee:' in line[rcol] and not is_senior_area_chair and not is_programme_chair
            is_reviewer = (
                line[rcol] != 'Author' and not line[rcol] == 'committee'
                and not is_senior_area_chair and not is_programme_chair and not is_area_chair
            )
            agreed_tscs = 'reviewer' in line[Rcol] or 'AC' in line[Rcol] or 'Yes' in line[Ecol]
            if is_reviewer and not agreed_tscs:
                print(f'WARNING: {line[ucol]} has not agreed to review or emergency review; role {line[rcol]}; agreed {line[Rcol]}')
                agreed_tscs = True

            # Check for not author, and not PC "manager:committee" and not SAC 
            # "committee:<track name> (manager #)"
            # If conditions are met, append the reviewer data to the reviewer output file
            if is_reviewer and agreed_tscs or is_area_chair or is_senior_area_chair:
                track = re.sub(r'committee:', '', line[rcol])
                track = re.sub(r':?Meta Reviewer:?', '', track).strip()
                track = re.sub(r'^:', '', track).strip()
                # if track != 'Computational Social Science and Social Media':
                #     continue
                if is_senior_area_chair:
                    # sigh, some track names contain the separator symbol ':'
                    track = re.sub(r'(.+):\1 \(manager.*', r'\1', track)
                emergency = ('Yes' in line[Ecol] and 'no' == line[Rcol])
                rev_data = {
                    'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id],
                    'startUsername': line[ucol], 'areaChair': is_area_chair, 'emergency': emergency, 
                    'track': track, 'seniorAreaChair': is_senior_area_chair
                }
                print(json.dumps(rev_data), file=f)
                reviewer_map[line[ucol]] = len(reviewers)
                reviewer_map[line[ecol]] = len(reviewers)
                reviewers.append(rev_data)

            # FIXME: experience / graduation year is also useful


    # ----------------------------------------------------------------------------------------------
    # Process the submission csv file and submission/reviewer match scores, factoring in COI
    # decisions and filtering out desk-rejections, cleaning and writing to the submission_out and
    # bid_out output files
    # ----------------------------------------------------------------------------------------------

    # Read-in submission profiles and process field IDs
    if not args.submission_in: 
        raise RuntimeError(f'Submission input file not included')
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = [
        'Submission ID', 'Title', 'Track', 'Submission Type', 'Abstract|Summary', 'Authors',
        'All Author Emails', 'Acceptance Status'
    ]
    scol, tcol, rcol, ycol, abscol, acol, ecol, stcol = find_colids(colnames, csv_lines[0])
    icols = find_colids([f'{i}: Username' for i in range(1, 99)], csv_lines[0])
    submissions, submission_map = [], {}

    # Load up SoftConf bids
    #   this can be straight from softconf "Bid_Information.csv"
    #   but better to use the reviewer-coi-detection code which adds additional COIs
    # The bids take the form of a matrix of compatibilities from submissions to reviewers, though
    # softconf and this codebase use different numerical conventions.
    # Softconf codes:
    #     1 = Yes, 2 = Maybe, 3 = No, 4 = COI
    # ACL codebase codes:
    #     0 = COI, 1 = No, 2 = Maybe, 3 = Yes
    # The matrix is initialized with 2s ("Maybes")
    bids = np.full( (len(csv_lines)-1, len(reviewers)), 2)

    # If there is a bid file from the COI module, load in the initializations
    if args.bid_in:
        bids_in = pandas.read_csv(
            args.bid_in, skipinitialspace=True, index_col='Submission ID/Username'
        )
        if bids_in.columns[-1].startswith("Unnamed:"):
            bids_in.drop(bids_in.columns[-1], axis=1, inplace=True)
        for i, (sid, sub_bids) in enumerate(bids_in.iterrows()):
            # process bids (even with no bidding, this still has COIs)
            for username, bidding_code in sub_bids.to_dict().items():
                if bidding_code in '1234':
                    reviewer_id = reviewer_map.get(username)
                    if reviewer_id != None:
                        # Change bid codes from softconf convention to ACL convention
                        bids[i, reviewer_id] = 4 - int(bidding_code)
                    else:
                        # this reviewer is an SAC or similar
                        assert username in profile_map
        
    # Loop over submissions to filter out rejected and invalid entries and write valid ones to the
    # output file. If an author email or id matches that of a reviewer, the bid code for that
    # submission -> reviewer combination is automatically set to 0 (COI)
    delim_re = re.compile(r'(?:, | and )')
    not_found = set()
    submission_kept = []
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]):
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            author_startids = [line[icols[j]] for j in range(len(author_emails))]
            authors = []
            # Loop over author profiles to make sure any reviewers among the authors do not get
            # assigned to their own submission
            for ae, an, ai in zip(author_emails, author_names, author_startids):
                aep = profile_map.get(ae)
                aip = profile_map.get(ai)
                if aep or aip:
                    authors.append(aep or aip)
                    aer = reviewer_map.get(ae)
                    air = reviewer_map.get(ai)
                    if aer != None:
                        bids[i,aer] = 0
                    elif air != None:
                        bids[i,air] = 0
                else:
                    print(f'WARNING: could not find account for {ae}, just using name {an}; username is "{ai}"')
                    authors.append({'name': an, 'ids': []})
                    if ai: not_found.add(ai)
            
            track = line[rcol]
            # BEGIN DELETE ME
            # Temporary fix to add n/a track where missing during script testing
            if not track:
                track = "n/a"
            # END DELETE ME
            assert track
            
            # Read in the submission type to either short or long, raising a value error if a
            # submission is of an invalid type
            if 'short' in line[ycol].lower():
                typ = 'short'
            elif 'long' in line[ycol].lower():
                typ = 'long'
            else:
                raise ValueError(f'Illegal Submission Type {line[ycol]}')
            # If the submission does not already have a reject status, write the submission
            # information to the output file and add the index to the list of submissions to be
            # kept
            if 'Reject' not in line[stcol]:
                data = {
                    'title': line[tcol], 'track': track, 'type': typ, 'paperAbstract': line[abscol], 
                    'authors': authors, 'startSubmissionId': line[scol]
                }
                submissions.append(data)
                #submission_map[line[scol]] = i
                print(json.dumps(data), file=f)
                submission_kept.append(i)

    # Keep the rows of the bid matrix that correspond to to the submissions that were not already
    # rejected and write to the bid_out file
    if args.bid_out:
        bids = bids[submission_kept, :]
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids)
