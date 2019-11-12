import argparse
import requests
import suggest_utils
import sys
import time

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

    reviewer_papers = set()
    rev_len = len(reviewer_ids)
    print(f'Querying s2 for {rev_len} reviewers (.=50 reviewers)', file=sys.stderr)
    sleep_time = 1
    for i, rid in enumerate(reviewer_ids):
        r = requests.get(f'http://api.semanticscholar.org/v1/author/{rid}')
        while r.status_code == 429:
            sleep_time *= 2
            print(f'WARNING: Hit rate limit. Increasing sleep to {sleep_time} ms', file=sys.stderr)
            time.sleep(sleep_time / 1000.0)
            r = requests.get(f'http://api.semanticscholar.org/v1/author/{rid}')
        if r.status_code != 200:
            print(f'WARNING: Could not access rid {rid}', file=sys.stderr)
        else:
            user = r.json()
            # print(json.dumps(user))
            for paper in user['papers']:
                reviewer_papers.add(paper['paperId'])
            suggest_utils.print_progress(i, 50)
        time.sleep(sleep_time / 1000.0)
    for x in reviewer_papers:
        print(x)

