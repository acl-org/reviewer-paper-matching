import argparse
import json
import numpy as np
import suggest_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--suggestion_file", type=str, required=True, help="A line-by-line JSON file of suggestions")
    parser.add_argument("--reviewer_file", type=str, required=True, help="A file of reviewer files or IDs that can review this time")
    parser.add_argument("--filter_field", type=str, default="Name", help="Which field to filter on")
    parser.add_argument("--bid_file", type=str, required=True, help="A file containing numpy array of bids (0 = COI, 1 = no, 2 = maybe, 3 = yes)")

    args = parser.parse_args()

    # Load the data
    with open(args.suggestion_file, "r") as f:
        submissions = [json.loads(x) for x in f]
    with open(args.reviewer_file, "r") as f:
        reviewer_aliases = [x.strip().split('|') for x in f]
    bids = np.load(args.bid_file)
    mapping = suggest_utils.calc_reviewer_db_mapping(reviewer_aliases, submissions, author_field='assignedReviewers')

    all_assignments = np.sum(mapping)

    # Total of all bid scores, minus one, divided by number of assignments
    for bid in range(4):
        bid_count = np.sum(np.where((mapping == 1) & (bids == bid), 1, 0))
        print(f'Ratio of {bid}: {bid_count/all_assignments}')

