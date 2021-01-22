import argparse
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

    # TEMPORARY FIX: assume each reviewer only belongs to the first track they
    # indicate
    primary_tracks = [tracks[0] for tracks in reviewers['tracks']]
    reviewers = reviewers.assign(tracks=primary_tracks)

    submissions = pd.read_json(args.submission_file, lines=True)
    quotas = pd.read_csv(args.quota_file)

    reviewer_usernames = list(reviewers['startUsername'])
    quota_usernames = list(quotas['Username'])
    quota_overlap = [name in reviewer_usernames for name in quota_usernames]
    quotas = quotas[quota_overlap].reset_index(drop=True)
    quotas['QuotaForReview'] = quotas['QuotaForReview'].fillna(args.default_paper_quota)
    quotas['QuotaForReview'] = quotas['QuotaForReview'].replace(to_replace='None', value=args.default_paper_quota)
    individual_quotas = {}
    for username, quota in zip(quotas['Username'], quotas['QuotaForReview']):
        individual_quotas[username] = int(quota)

    # Get the set of tracks to iterate over
    tracks = list(set(reviewers['tracks']))

    # Sort the reviewers into tables by their primary track
    for track in tracks:
        track_reviewers = reviewers[reviewers['tracks'] == track]
        track_submissions = submissions[submissions['track'] == track]
        print(f'Track name: {track}')
        print(f'Number of reviewers: {len(track_reviewers)}')
        print(f'Number of papers: {len(track_submissions)}')
        
        if args.quota_file:
            quota_sum = 0
            for username in track_reviewers['startUsername']:
                quota_sum += individual_quotas[username]
        else:
            quota_sum = args.default_paper_quota * len(track_reviewers)
        print(f'Number reviewer-paper assignments available: {quota_sum}')
        
        papers_times_committee = args.reviewers_per_paper * len(track_submissions)
        print(f'Number reviewer-paper assignments needed: {papers_times_committee}')

        reviewer_info = list(zip(track_reviewers['name'], track_reviewers['startUsername']))

        print('Reviewer info:')
        for reviewer in reviewer_info:
            print(f'{reviewer[0]}, {reviewer[1]}')
        print('\n\n')


if __name__ == '__main__':
    main()
