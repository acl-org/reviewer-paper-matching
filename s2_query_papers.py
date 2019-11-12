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

    parser.add_argument("--paperid_file", type=str, required=True, help="A text file of paper IDs to match")
    parser.add_argument("--db_file", type=str, default=None, help="A JSON file of already-cached papers")

    args = parser.parse_args()

    with open(args.paperid_file, "r") as f:
        paper_ids = set([line.strip() for line in f])

    if args.db_file:
        pid_len = len(paper_ids)
        print(f'Getting {pid_len} papers from dump if they exist (.=1000 papers)', file=sys.stderr)
        with open(args.db_file, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if data['id'] in paper_ids:
                    paper_ids.remove(data['id'])
                    print(line.strip())
                suggest_utils.print_progress(i, 1000)
        print('', file=sys.stderr)

    pid_len = len(paper_ids)
    print(f'Querying s2 for {pid_len} remaining papers (.=50 papers)', file=sys.stderr)
    for i, pid in enumerate(paper_ids):
        r = requests.get(f'http://api.semanticscholar.org/v1/paper/{pid}')
        pmap = r.json()
        pmap = {v: pmap[k] for (k, v) in paper_info.items()}
        for i, auth in enumerate(pmap['authors']):
            pmap['authors'][i] = {'name': auth['name'], 'ids': [auth['authorId']]}
        print(json.dumps(pmap))
        suggest_utils.print_progress(i, 50)


