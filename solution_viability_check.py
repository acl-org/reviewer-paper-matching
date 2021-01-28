import argparse
import itertools
import pandas as pd

def main():
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
    quotas['QuotaForReview'].replace(
        to_replace='None', value=0, inplace=True
    )

    individual_quotas = {}
    for username, quota in zip(quotas['Username'], quotas['QuotaForReview']):
        individual_quotas[username] = int(quota)

    # Get the set of tracks to iterate over
    tracks = list(itertools.chain.from_iterable(reviewers['tracks']))
    tracks = set(tracks)

    # For each track, count reviewers, ACs, SACs, and submissions, calculating
    # the number of reviewers needed and printint any warnings
    for track in tracks:
        
        # Reviewers may have more than one track, so the track reviewers are
        # reviewers that have the current track in their track list
        belongs_to_track = [track in lst for lst in reviewers['tracks']]
        track_reviewers = reviewers[belongs_to_track]
        num_reviewers = len(track_reviewers)

        # Reviewers in only this track vs ones who are also in other tracks
        only_this_track = [
            len(lst) == 1 for lst in track_reviewers['tracks']
        ]
        reviewers_only_in_this_track = track_reviewers[only_this_track]
        num_reviewers_only_in_this_track = len(reviewers_only_in_this_track)
        num_reviewers_in_other_track = (
            num_reviewers - num_reviewers_only_in_this_track
        )
        
        # Submissions only have one track, so it suffices to gather them by the
        # simple value of their track field
        track_submissions = submissions[submissions['track'] == track]

        # ACs
        ac_belongs_to_track = [track in lst for lst in all_reviewers['ac_tracks']]
        track_acs = all_reviewers[ac_belongs_to_track]

        # SACs
        sac_belongs_to_track = [track in lst for lst in all_reviewers['sac_tracks']]
        track_sacs = all_reviewers[sac_belongs_to_track]
        
        # Print out basic stats about the track
        print(f'Track name: {track}')
        print(f'Number of papers: {len(track_submissions)}')
        print(f'Number of reviewers: {num_reviewers}')
        print(f'Number of reviewers in other tracks: {num_reviewers_in_other_track}')
        print(f'Number of ACs: {len(track_acs)}')
        print(f'Number of SACs: {len(track_sacs)}')

        # Calculate the total number of reviewer-paper assignments available,
        # based on the individual quota file if there is one, defaulting to the
        # provided default quota. Also calculate the number if only the
        # reviewers dedicated to this track count
        if args.quota_file:
            quota_sum = 0
            dedicated_quota_sum = 0
            for username in track_reviewers['startUsername']:
                quota_sum += individual_quotas[username]
            for username in reviewers_only_in_this_track['startUsername']:
                dedicated_quota_sum += individual_quotas[username]
        else:
            quota_sum = args.default_paper_quota * num_reviewers
            dedicated_quota_sum = (
                args.default_paper_quota * num_reviewers_only_in_this_track
            )
        print(f'Number reviewer-paper assignments available: {quota_sum}')
        print(f'Number reviewer-paper assignment dedicated to this track: {dedicated_quota_sum}')
        
        # Calculate the number of reviewer-paper assignments needed (based on
        # input number of reviewers per paper) and print a warning if the needed
        # number is greater than the number available
        papers_times_committee = (
            args.reviewers_per_paper * num_reviewers
        )
        print(f'Number reviewer-paper assignments needed: {papers_times_committee}')
        if papers_times_committee > quota_sum:
            print("Warning: number of reviews required exceeds number available")
        else:
            print("Warning: None")

        #
        reviewer_info = list(zip(
            track_reviewers['name'], track_reviewers['startUsername']
        ))
        print('Reviewer info:')
        for reviewer in reviewer_info:
            print(f'{reviewer[0]}, {reviewer[1]}, {individual_quotas[reviewer[1]]}')
        
        print('\n')


if __name__ == '__main__':
    main()
