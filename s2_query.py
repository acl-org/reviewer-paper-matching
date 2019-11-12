import argparse
import itertools
import requests
import json
import numpy as np
import suggest_utils
import sys

paper_info = {"paperId": "id", "title": "title", "abstract": "paperAbstract", "year": "year", "authors": "authors", "venue": "venue"}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--reviewer_file", type=str, required=True, help="A tsv file of reviewer names and IDs to query")

    args = parser.parse_args()

    with open(args.reviewer_file, "r") as f:
        reviewer_data = [[y.split('|') for y in x.strip().split('\t')] for x in f]
        reviewer_ids = set()
        for x in reviewer_data:
            for myid in x[1]:
                reviewer_ids.add(myid)

    retrieved_papers = {}
    for rid in reviewer_ids:
        r = requests.get(f'http://api.semanticscholar.org/v1/author/{rid}')
        if r.status_code != 200:
            raise ValueError(f'Could not access rid {rid}')
        user = r.json()
        # print(json.dumps(user))
        for paper in user['papers']:
            pid = paper['paperId']
            if pid not in retrieved_papers:
                r = requests.get(f'http://api.semanticscholar.org/v1/paper/{pid}')
                pmap = r.json()
                pmap = {v: pmap[k] for (k, v) in paper_info.items()}
                for i, auth in enumerate(pmap['authors']):
                    pmap['authors'][i] = {'name': auth['name'], 'ids': [auth['authorId']]}
                print(json.dumps(pmap))
                retrieved_papers[pid] = 1


