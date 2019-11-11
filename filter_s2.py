import argparse
import json
import numpy as np
import suggest_utils
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--reviewer_file", type=str, required=True, help="A tsv file of reviewer names and IDs")
    parser.add_argument("--filter_field", type=str, default="name", help="Which field to filter on (name,id)")

    args = parser.parse_args()

    # Load the data
    with open(args.reviewer_file, "r") as f:
        reviewer_data = [[y.split('|') for y in x.strip().split('\t')] for x in f]
        reviewer_names = [x[0][0] for x in reviewer_data]
    mapping = suggest_utils.calc_reviewer_id_mapping(reviewer_data, args.filter_field)

    for line in sys.stdin:
        data = json.loads(line)
        if args.filter_field == 'id':
            ok = any([any([(y in mapping) for y in x['ids']]) for x in data['authors']])
        else:
            ok = any([(x['name'] in mapping) for x in data['authors']])
        if ok:
            print(line.strip())
