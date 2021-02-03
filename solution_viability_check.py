import argparse
import itertools
import pandas as pd


def main():
    """
    A script to check whether the paper assignment problem is solvable (i.e.,
    if there are enough reviewers within each track to review all of the track
    papers)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--reviewer_file',
        type=str,
        required=True,
        help='The jsonl reviewer file created by softconf_extract.py'
    )
    parser.add_argument(
        '--submission_file',
        type=str,
        required=True,
        help='The jsonl submission file created by softconf_extract.py'
    )
    parser.add_argument(
        '--default_paper_quota',
        type=int,
        required=True,
        help='The default number of papers to be assigned to each reviewer'
    )
    parser.add_argument(
        '--default_ac_quota',
        type=int,
        default=15,
        help='The default number of papers to be assigned to each area chair'
    )
    parser.add_argument(
        '--reviewers_per_paper',
        type=int,
        required=True,
        help='The number of reviewers to be assigned to each paper'
    )
    parser.add_argument(
        '--quota_file',
        type=str,
        help=(
            'A csv file downloaded from softconf containing reviewer IDs'
            ' and their paper quota'
        )
    )

    args = parser.parse_args()

    # Read in the reviewer file from jsonl format
    all_reviewers = pd.read_json(args.reviewer_file, lines=True)

    # Some of the entries will have an empty list under the "tracks" column,
    # due to the fact that they are ACs or SACs who are not reviewing in
    # separate track. Here we filter away those ones
    has_review_track = [len(tracks) > 0 for tracks in all_reviewers['tracks']]
    reviewers = all_reviewers[has_review_track].reset_index(drop=True)

    submissions = pd.read_json(args.submission_file, lines=True)
    quotas = pd.read_csv(args.quota_file)

    reviewer_usernames = list(reviewers['startUsername'])
    quota_usernames = list(quotas['Username'])
    quota_overlap = [name in reviewer_usernames for name in quota_usernames]
    quotas = quotas[quota_overlap].reset_index(drop=True)
    quotas['QuotaForReview'].fillna(args.default_paper_quota, inplace=True)
    quotas['QuotaForReview'].replace(to_replace='None', value=0, inplace=True)

    individual_quotas = {}
    for username, quota in zip(quotas['Username'], quotas['QuotaForReview']):
        individual_quotas[username] = int(quota)

    # Right now the reviewer assignment script prevents ACs and SACs being
    # assigned as normal reviewers
    for i in range(len(reviewers)):
        if len(
            reviewers.loc[i, 'ac_tracks'] + reviewers.loc[i, 'sac_tracks']
        ) > 0:
            individual_quotas[reviewers.loc[i, 'startUsername']] = 0

    # Get the set of tracks to iterate over
    reviewer_tracks = list(itertools.chain.from_iterable(reviewers['tracks']))
    paper_tracks = list(submissions['track'])
    tracks = set(reviewer_tracks) | set(paper_tracks)

    # For each track, count reviewers, ACs, SACs, and submissions, calculating
    # the number of reviewers needed and printint any warnings
    for track in tracks:

        # Reviewers may have more than one track, so the track reviewers are
        # reviewers that have the current track in their track list
        belongs_to_track = [track in lst for lst in reviewers['tracks']]
        track_reviewers = reviewers[belongs_to_track].reset_index(drop=True)
        num_reviewers = len(track_reviewers)

        # Reviewers in only this track vs ones who are also in other tracks
        only_this_track = [
            len(lst) == 1 for lst in track_reviewers['tracks'] +
            track_reviewers['ac_tracks'] + track_reviewers['sac_tracks']
        ]
        reviewers_only_in_this_track = track_reviewers[only_this_track]
        num_reviewers_only_in_this_track = len(reviewers_only_in_this_track)
        num_reviewers_in_other_track = (
            num_reviewers - num_reviewers_only_in_this_track
        )

        # Submissions only have one track, so it suffices to gather them by the
        # simple value of their track field
        track_submissions = submissions[submissions['track'] == track]
        num_papers = len(track_submissions)

        # ACs
        ac_belongs_to_track = [
            track in lst for lst in all_reviewers['ac_tracks']
        ]
        track_acs = all_reviewers[ac_belongs_to_track].reset_index(drop=True)
        num_acs = len(track_acs)

        # SACs
        sac_belongs_to_track = [
            track in lst for lst in all_reviewers['sac_tracks']
        ]
        track_sacs = all_reviewers[sac_belongs_to_track].reset_index(drop=True)

        # Print out basic stats about the track
        print(f'Track name: {track}')
        print(f'Number of papers: {num_papers}')
        print(f'Number of reviewers: {num_reviewers}')
        print(
            f'Number of reviewers in other tracks: {num_reviewers_in_other_track}'
        )
        print(f'Number of ACs: {num_acs}')
        print(f'Number of SACs: {len(track_sacs)}')

        # If you wanted to give each AC and individual quota, the code would
        # need to be changed
        ac_quota_sum = args.default_ac_quota * num_acs
        if num_acs > 0:
            papers_per_ac = num_papers / num_acs
        else:
            papers_per_ac = float('inf')
        print(f'Number of papers per AC: {round(papers_per_ac, 1)}')
        print(f'Number AC assignments available: {ac_quota_sum}')
        if num_papers > ac_quota_sum:
            print(
                "AC Warning: number of papers exceeds number of AC assignments available"
            )
        else:
            print("AC Warning: None")

        # Calculate the total number of reviewer-paper assignments available,
        # based on the individual quota file if there is one, defaulting to the
        # provided default quota. Also calculate the number if only the
        # reviewers dedicated to this track count
        if args.quota_file:
            quota_sum = 0
            dedicated_quota_sum = 0
            if num_reviewers > 0:
                for username in track_reviewers['startUsername']:
                    quota_sum += individual_quotas[username]
            if num_reviewers_only_in_this_track > 0:
                for username in reviewers_only_in_this_track['startUsername']:
                    dedicated_quota_sum += individual_quotas[username]
        else:
            quota_sum = args.default_paper_quota * num_reviewers
            dedicated_quota_sum = (
                args.default_paper_quota * num_reviewers_only_in_this_track
            )
        print(f'Number reviewer-paper assignments available: {quota_sum}')
        print(
            f'Number reviewer-paper assignments dedicated to this track: {dedicated_quota_sum}'
        )

        # Calculate the number of reviewer-paper assignments needed (based on
        # input number of reviewers per paper) and print a warning if the needed
        # number is greater than the number available
        papers_times_committee = (args.reviewers_per_paper * num_papers)
        if num_reviewers > 0:
            papers_per_reviewer = num_papers / num_reviewers
        else:
            papers_per_reviewer = float('inf')
        print(f'Number of papers per reviewer: {round(papers_per_reviewer, 1)}')
        print(
            f'Number reviewer-paper assignments needed: {papers_times_committee}'
        )
        if papers_times_committee > quota_sum:
            print(
                "Reviewer Warning: number of reviews required exceeds number available"
            )
        else:
            print("Reviewer Warning: None")

        print('Area Chair Info:')
        print('  username, name, other ac tracks, reviewing tracks')
        for i in range(num_acs):
            username = track_acs.loc[i, 'startUsername']
            name = track_acs.loc[i, 'name']
            other_ac_tracks = ';'.join(
                set(track_acs.loc[i, 'ac_tracks']) - set([track])
            )
            reviewing_tracks = ';'.join(track_acs.loc[i, 'tracks'])
            info = ', '.join(
                [username, name, other_ac_tracks, reviewing_tracks]
            )
            print(f'  {info}')

        print('Reviewer info:')
        print(
            '  username, name, quota, other reviewing tracks, ac tracks, sac tracks'
        )
        for i in range(num_reviewers):
            username = track_reviewers.loc[i, 'startUsername']
            name = track_reviewers.loc[i, 'name']
            quota = str(individual_quotas[username])
            other_tracks = ';'.join(
                set(track_reviewers.loc[i, 'tracks']) - set([track])
            )
            ac_tracks = ';'.join(track_reviewers.loc[i, 'ac_tracks'])
            sac_tracks = ';'.join(track_reviewers.loc[i, 'ac_tracks'])
            info = [username, name, quota, other_tracks, ac_tracks, sac_tracks]
            info = ', '.join(info)
            print(f'  {info}')
        print('')


if __name__ == '__main__':
    main()
