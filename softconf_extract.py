import argparse
import csv
import json
import re
import sys
import warnings
import pandas
import suggest_utils
import numpy as np


def find_colids(colnames, header):
    """Get the csv column index of each field in the colnames parameter"""
    colids, colmap = [-1 for _ in colnames], {}
    for i, name in enumerate(colnames):
        for alias in name.split('|'):
            colmap[alias] = i
    for i, col in enumerate(header):
        if col in colmap:
            colids[colmap[col]] = i
    if any([x == -1 for x in colids]):
        print(
            f'WARNING: Couldn\'t find column ids in {colnames} {colids}',
            file=sys.stderr
        )
    return colids


def is_area_chair_from(role):
    """
    Compute if the role description corresponds to a area chair, and also return
    the track(s) in which they are an area chair

    Usually, ACs have 'Meta Reviewer" in their role description,
    e.g. "committee:Summarization:Meta Reviewer"
    """
    if 'Meta Reviewer' in role:
        no_colon_role = re.sub(r': ', '- ', role)
        track_strings = re.findall(r'(.+):Meta Reviewer$', no_colon_role)
        if len(track_strings) > 0:
            track_strings = track_strings[0].split(':')
        else:
            track_strings = ['']
        return (True, track_strings)
    else:
        return (False, [])


def is_senior_area_chair_from(role):
    """ 
    Computes if the role description corresponds to a senior area chair, and
    also return the track(s) in which they are an area chair

    SACs are discinguishable from everyone else via the "manager" suffix.
    E.g. committee:Speech:Speech (manager 1)
    """
    if '(manager' in role:
        no_colon_role = re.sub(r': ', '- ', role)
        track_strings = re.findall('([^:]+) \(manager', no_colon_role)
        if len(track_strings) > 0:
            track_strings = [track.strip() for track in track_strings]
        else:
            track_strings = ['']
        return (True, track_strings)
    else:
        return (False, [])


def is_reviewer_from(role):
    """
    Computes if the role description corresponds to a reviewer

    Before reviewer track assignment each reviewer has the role 'committee'
    and after - most of the reviewers have new format: 'committee:TRACK'
    where TRACK is the track name (note that track names may contain ':'-symbol)

    Note that if you do not want to assign papers to the reviewers without
    track, you can provide --track argument to the suggest_reviewers.py

    DO NOT MODIFY THIS FUNCTION UNLESS YOU FULLY UNDERSTAND THIS REPOSITORY.
    """
    if role == 'committee':
        return True

    if role.startswith(
        'committee:'
    ) and not is_senior_area_chair_from(role) and not is_area_chair_from(role):
        return True

    # BEGIN TEMPORARY FIX
    # For ACL-2021, softconf lists reviewer roles simply as the name of the
    # track. "committee" does not appear in the field
    if role != "Author" and not is_senior_area_chair_from(role)[
        0] and not is_area_chair_from(role)[0]:
        return True
    # END TEMPORARY FIX

    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submission_in",
        type=str,
        required=True,
        help="The submission CSV file"
    )
    parser.add_argument(
        "--profile_in", type=str, required=True, help="The profile CSV file"
    )
    parser.add_argument(
        "--bid_in", type=str, default=None, help="The bids CSV file"
    )
    parser.add_argument(
        "--reviewer_out",
        type=str,
        required=True,
        help="The cleaned reviewer jsonl file"
    )
    parser.add_argument(
        "--submission_out",
        type=str,
        default=None,
        help="The cleaned submission jsonl file"
    )
    parser.add_argument(
        "--bid_out",
        type=str,
        default=None,
        help="A numpy matrix of submission/reviewer scores"
    )
    parser.add_argument(
        "--reject_out",
        type=str,
        default=None,
        help="An output file listing rejected submissions"
    )
    parser.add_argument(
        "--non_reviewers_out",
        type=str,
        default=None,
        help="An output file listing non-reviewers"
    )
    parser.add_argument(
        "--multi_track_reviewers_out",
        type=str,
        default=None,
        help="An output file listing reviewers signed up for multiple tracks"
    )

    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    # --------------------------------------------------------------------------
    # Process reviewer/author profile csv file from softconf and clean for
    # output file
    # --------------------------------------------------------------------------

    # Read-in reviewer and author profiles and process field IDs
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        profiles_csv = list(csv_reader)

    colnames = [
        'Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID',
        'Roles'
    ]
    ucol, ecol, fcol, lcol, scol, rcol = find_colids(colnames, profiles_csv[0])

    # Also check that they've agreed to review in the local profile questions
    Rcol, Ecol = find_colids(['PCRole', 'emergencyReviewer'], profiles_csv[0])
    last_field = max([ucol, ecol, fcol, lcol, scol, rcol, Rcol, Ecol])
    reviewers, reviewer_map, profile_map = [], {}, {}

    multi_track_reviewer_data = []
    non_reviewer_data = []

    # Loop through reviewer and author profiles
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(profiles_csv[1:]):

            # Some lines are incomplete due to the rightward columns being blank
            if len(line) < last_field + 1:
                line.extend([''] * (last_field + 1 - len(line)))

            # FIXME: is there better parsing of these from the other repository?
            s2id = line[scol].split('/')[-1]

            # Map author/reviewer username and email to basic profile info
            data = {
                'name': f'{line[fcol]} {line[lcol]}',
                'ids': [s2id],
                'startUsername': line[ucol]
            }
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data

            # Check for special reviewer roles, and whether they have agreed to
            # emergency review
            _role = line[rcol]

            is_area_chair, ac_tracks = is_area_chair_from(_role)
            is_senior_area_chair, sac_tracks = is_senior_area_chair_from(_role)
            is_programme_chair = _role == "manager:committee"
            is_reviewer = is_reviewer_from(_role)

            # ACL 2021: Commenting out this block because it doesn't make much
            # sense
            # agreed_tscs = 'reviewer' in line[Rcol] or 'AC' in line[Rcol] or 'Yes' in line[Ecol]
            # if is_reviewer[0] and not agreed_tscs:
            # print(
            #     f"WARNING: {line[ucol]} has not agreed to review or emergency"
            #     " review; role {_role}; agreed {line[Rcol]}"
            # )
            # This line below does not make sense
            # agreed_tscs = True

            # Check for not author, and not PC "manager:committee" and not SAC
            # "committee:<track name> (manager #)"
            # If conditions are met, append the reviewer data to the reviewer 
            # output file
            if is_reviewer or is_area_chair or is_senior_area_chair:
                
                # This line may be outdated, since 'committee' no longer appears
                # at the beginning of the reviewer role value
                track = re.sub(r'committee:', '', _role)
                track = re.sub(r':?Meta Reviewer:?', '', track).strip()
                track = re.sub(r'^:', '', track).strip()
                
                # Get rid of the manager parenthetical in the track name
                track = re.sub(r' \(manager ?[0-9]?\)', '', track)
                
                # Some tracks contain a colon within them, though it is always
                # followed by a space. For the purposes of being able to split
                # up multiple tracks by colons, it is changed to a dash
                track = re.sub(r': ', '- ', track)
                
                # Reviewers may have multiple tracks, so we will split on colons
                tracks = track.split(':')
                
                # Compute the (reviewer) tracks as the set difference between
                # all tracks and the reviewer's AC and SAC tracks
                reviewer_track_set = set(tracks)
                ac_tracks = [
                    track for track in ac_tracks if ' (manager' not in track
                ]
                ac_track_set = set(ac_tracks)
                sac_track_set = set(sac_tracks)
                tracks = list(reviewer_track_set - ac_track_set - sac_track_set)
                num_tracks = len(tracks) + len(ac_tracks) + len(sac_tracks)
                emergency = ('Yes' in line[Ecol] and 'no' == line[Rcol])
                rev_data = {
                    'name': f'{line[fcol]} {line[lcol]}',
                    'ids': [s2id],
                    'startUsername': line[ucol],
                    'tracks': tracks,
                    'areaChair': is_area_chair,
                    'ac_tracks': ac_tracks,
                    'seniorAreaChair': is_senior_area_chair,
                    'sac_tracks': sac_tracks,
                    'emergency': emergency
                }
                print(json.dumps(rev_data), file=f)
                reviewer_map[line[ucol]] = len(reviewers)
                reviewer_map[line[ecol]] = len(reviewers)
                reviewers.append(rev_data)
                if args.multi_track_reviewers_out and num_tracks > 1:
                    data = {
                        'start_id': line[ucol],
                        'email': line[ecol],
                        'name': f'{line[fcol]} {line[lcol]}',
                        'reviewer_tracks': tracks,
                        'ac_tracks': ac_tracks,
                        'sac_tracks': sac_tracks,
                        'softconf_role_string': _role
                    }
                    multi_track_reviewer_data.append(data)
            elif args.non_reviewers_out:
                track = _role
                emergency = ('Yes' in line[Ecol] and 'no' == line[Rcol])
                data = {
                    'name': f'{line[fcol]} {line[lcol]}',
                    'ids': [s2id],
                    'startUsername': line[ucol],
                    'areaChair': is_area_chair,
                    'emergency': emergency,
                    'tracks': track,
                    'seniorAreaChair': is_senior_area_chair
                }
                non_reviewer_data.append(data)

            # FIXME: experience / graduation year is also useful

    if args.multi_track_reviewers_out:
        with open(args.multi_track_reviewers_out, 'w+') as f:
            for line in multi_track_reviewer_data:
                print(json.dumps(line), file=f)

    if args.non_reviewers_out:
        with open(args.non_reviewers_out, 'w+') as f:
            for line in non_reviewer_data:
                print(json.dumps(line), file=f)

    # --------------------------------------------------------------------------
    # Process the submission csv file and submission/reviewer match scores,
    # factoring in COI decisions and filtering out desk-rejections, cleaning and
    # writing to the submission_out and bid_out output files
    # --------------------------------------------------------------------------

    # Read-in submission profiles and process field IDs
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        submissions_csv = list(csv_reader)
    colnames = [
        'Submission ID', 'Title', 'Track', 'Submission Type',
        'Abstract|Summary', 'Authors', 'All Author Emails', 'Acceptance Status'
    ]
    scol, tcol, rcol, ycol, abscol, acol, ecol, stcol = find_colids(
        colnames, submissions_csv[0]
    )
    icols = find_colids(
        [f'{i}: Username' for i in range(1, 99)], submissions_csv[0]
    )
    submissions, submission_map = [], {}

    # Load up SoftConf bids
    # This can be straight from softconf "Bid_Information.csv", but better to
    # use the reviewer-coi-detection code which adds additional COIs
    # The bids take the form of a matrix of compatibilities from submissions to
    # reviewers, though softconf and this codebase use different numerical
    # conventions.
    # Softconf codes:
    #     1 = Yes, 2 = Maybe, 3 = No, 4 = COI
    # ACL codebase codes:
    #     0 = COI, 1 = No, 2 = Maybe, 3 = Yes
    # The matrix is initialized with 2s ("Maybes")
    bids = np.full((len(submissions_csv) - 1, len(reviewers)), 2)

    # If there is a bid file from the COI module, load in the initializations
    if args.bid_in:
        bids_in = pandas.read_csv(
            args.bid_in, skipinitialspace=True, index_col=0
        )
        if bids.shape[0] != bids_in.shape[0]:
            raise RuntimeError(
                "--bids_in should have the rows corresponding to the rows in"
                " the --submission_in"
            )
        if bids_in.columns[-1].startswith("Unnamed:"):
            bids_in.drop(bids_in.columns[-1], axis=1, inplace=True)
        for i, (sid, sub_bids) in enumerate(bids_in.iterrows()):
            # process bids (even with no bidding, this still has COIs)
            for username, bidding_code in sub_bids.to_dict().items():
                if isinstance(
                    bidding_code, float
                ) and bidding_code != bidding_code:
                    warnings.warn(
                        f"NaN value found for username {username}", UserWarning
                    )
                    continue
                if bidding_code in '1234':
                    reviewer_id = reviewer_map.get(username)
                    if reviewer_id != None:
                        # Change bid codes from softconf convention to ACL
                        # convention
                        bids[i, reviewer_id] = 4 - int(bidding_code)

    # Loop over submissions to filter out rejected and invalid entries and write
    # valid ones to the output file. If an author email or id matches that of a
    # reviewer, the bid code for that submission -> reviewer combination is
    # automatically set to 0 (COI)
    delim_re = re.compile(r'(?:, | and )')
    not_found = set()
    submission_kept = []
    reject_data = []
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(submissions_csv[1:]):
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            author_startids = [
                line[icols[j]] for j in range(len(author_emails))
            ]
            authors = []
            
            # Loop over author profiles to make sure any reviewers among the
            # authors do not get assigned to their own submission
            for ae, an, ai in zip(author_emails, author_names, author_startids):
                aep = profile_map.get(ae)
                aip = profile_map.get(ai)
                if aep or aip:
                    authors.append(aep or aip)
                    aer = reviewer_map.get(ae)
                    air = reviewer_map.get(ai)
                    if aer != None:
                        bids[i, aer] = 0
                    elif air != None:
                        bids[i, air] = 0
                else:
                    warnings.warn(
                        f"Could not find account for {ae}, just using name" 
                        f" {an}; username is '{ai}'",
                        RuntimeWarning
                    )
                    authors.append({'name': an, 'ids': []})
                    if ai:
                        not_found.add(ai)

            track = line[rcol]
            track = re.sub(r':', '-', track)

            # Read in the submission type to either short or long, raising a
            # value error if a
            # submission is of an invalid type
            if 'short' in line[ycol].lower():
                typ = 'short'
            elif 'long' in line[ycol].lower():
                typ = 'long'
            else:
                raise ValueError(f'Illegal Submission Type {line[ycol]}')
            
            # If the submission does not already have a reject status, write the
            # submission information to the output file and add the index to the
            # list of submissions to be kept
            if 'Reject' not in line[stcol]:
                if not track:
                    track = "n/a"
                    warnings.warn(
                        f"Submission ID {line[scol]} does not have a track"
                        " assigned.\n"
                        "  It will be set as 'n/a' for output, but it will not"
                        " be assigned any reviewers unless there are reviewers"
                        " in a track called 'n/a'",
                        UserWarning
                    )
                data = {
                    'title': line[tcol],
                    'track': track,
                    'type': typ,
                    'paperAbstract': line[abscol],
                    'authors': authors,
                    'startSubmissionId': line[scol]
                }
                print(json.dumps(data), file=f)
                submission_kept.append(i)
            else:
                data = {
                    "startSubmissionId": line[scol],
                    'track': track,
                    'title': line[tcol],
                    'status': line[stcol]
                }
                reject_data.append(data)
            
    if args.reject_out:
        with open(args.reject_out, 'w') as f:
            for line in reject_data:
                print(json.dumps(line), file=f)

    # Keep the rows of the bid matrix that correspond to to the submissions that
    # were not already rejected and write to the bid_out file
    if args.bid_out:
        bids = bids[submission_kept, :]
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids)


if __name__ == "__main__":
    main()
