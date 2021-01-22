import argparse
import pandas as pd

main():
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
    
    args = parser.parse_args()

    # Read in the reviewer file from jsonl format
    all_reviewers = pd.read_json(args.reviewer_file, lines=True)

    # Some of the entries will have an empty list under the "tracks" column,
    # due to the fact that they are ACs or SACs who are not reviewing in
    # separate track. Here we filter away those ones
    has_review_track = [len(tracks) > 0 for tracks in all_reviewers['tracks']]
    reviewers = all_reviewers[has_review_track]

    # TEMPORARY FIX: assume each reviewer only belongs to the first track they
    # indicate
    primary_tracks = [tracks[0] for tracks in reviewers['tracks']]
    reviewers = reviewers.assign(tracks=primary_tracks)

    submissions = pd.read_json(args.submission_file)

    # Get the set of tracks to iterate over
    tracks = list(set(reviewers['tracks']))

    # Sort the reviewers into tables by their primary track
    for track in tracks:
        track_reviewers = reviewers[reviewers['tracks'] == track]
        track_submissions = submissions[submissions['track'] == track]
        print(f'Track name: {track}')
        print(f'Number of reviewers: {len(track_reviewers)}')
        print(f'Number of papers: {len(track_submissions)}')
        quota_times_reviewers = args.default_paper_quota * len(track_reviewers)
        print(f'Number reviewer-paper assignments available: {quota_times_reviewers}')
        papers_times_committee = args.reviewers_per_paer * len(track_submissions)
        print(f'Number reviewer-paper assignments needed: {papers_times_committee}')

        reviewer_info = list(zip(track_reviewers['name'], track_reviewers['startUsername']))

        print('Reviewer info:')
        for reviewer in reviewer_info:
            print(f'{reviewer[0]}, {reviewer[1]}')


if __name__ == '__main__':
    main()