import argparse
import json

import requests
import sys
import time

from tqdm import tqdm  # Progress bar

paper_info = {
    "paperId": "id",
    "title": "title",
    "abstract": "paperAbstract",
    "year": "year",
    "authors": "authors",
    "venue": "venue",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reviewer_file",
        type=str,
        required=True,
        help="A json file of reviewer names and IDs to query",
    )

    args = parser.parse_args()

    with open(args.reviewer_file, "r") as f:
        reviewer_data = [json.loads(line) for line in f]
        reviewer_ids = set()
        for x in reviewer_data:
            for myid in x["ids"]:
                reviewer_ids.add(myid)

    reviewer_papers = set()
    rev_len = len(reviewer_ids)
    print(f"Querying s2 for {rev_len} reviewers (.=50 reviewers)", file=sys.stderr)
    sleep_time = 1
    with requests.Session() as session:
        session.verify = False  # Not verifying SSL cert speeds things up.
        for i, rid in tqdm(enumerate(reviewer_ids)):
            r = session.get(f"http://api.semanticscholar.org/v1/author/{rid}")
            while r.status_code == 429:
                sleep_time *= 2
                print(
                    f"WARNING: Hit rate limit. Increasing sleep to {sleep_time} ms",
                    file=sys.stderr,
                )
                time.sleep(sleep_time / 1000.0)
                r = session.get(f"http://api.semanticscholar.org/v1/author/{rid}")
            if r.status_code != 200:
                print(f"WARNING: Could not access rid {rid}", file=sys.stderr)
            else:
                user = r.json()
                # print(json.dumps(user))
                for paper in user["papers"]:
                    reviewer_papers.add(paper["paperId"])
            time.sleep(sleep_time / 1000.0)
    for x in reviewer_papers:
        print(x)
