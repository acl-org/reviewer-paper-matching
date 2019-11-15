import argparse
import json
import csv
import suggest_utils
import sys
import re


def find_colids(colnames, header):
    colids = {k: -1 for k in colnames}
    for i, col in enumerate(header):
        if col in colids:
            colids[col] = i
    if any([x == -1 for x in colids.values()]):
        raise ValueError(f"Couldn't find column ids in {colids}")
    return [colids[x] for x in colnames]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reviewer_in", type=str, required=True, help="The reviewer CSV file"
    )
    parser.add_argument(
        "--submission_in", type=str, default=None, help="The submission CSV file"
    )
    parser.add_argument(
        "--bid_in", type=str, default=None, help="The submission CSV file"
    )
    parser.add_argument(
        "--reviewer_out", type=str, required=True, help="The reviewer CSV file"
    )
    parser.add_argument(
        "--submission_out", type=str, default=None, help="The submission CSV file"
    )
    parser.add_argument(
        "--bid_out", type=str, default=None, help="The submission CSV file"
    )

    args = parser.parse_args()

    # Process reviewers
    with open(args.reviewer_in, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        csv_lines = list(csv_reader)
    colnames = ["Username", "First Name", "Last Name", "Semantic Scholar ID"]
    ucol, fcol, lcol, scol = find_colids(colnames, csv_lines[0])
    reviewers, reviewer_map = [], {}
    with open(args.reviewer_out, "w") as f:
        for i, line in enumerate(csv_lines[1:]):
            s2id = line[scol].split("/")[-1]
            data = {
                "names": [f"{line[fcol]} {line[lcol]}"],
                "ids": [s2id],
                "username": line[ucol],
            }
            print(json.dumps(data), file=f)
            reviewers.append(data)
            reviewer_map[line[ucol]] = i

    # Process submissions (if present)
    if not args.submission_in:
        sys.exit(0)
    with open(args.submission_in, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        csv_lines = list(csv_reader)
    colnames = ["Submission ID", "Title", "Abstract", "Authors"]
    scol, tcol, abscol, acol = find_colids(colnames, csv_lines[0])
    delim_split = r"(?:, | and)"
    submissions, submission_map = [], {}
    with open(args.submission_out, "w") as f:
        for i, line in enumerate(csv_lines[1:]):
            authors = re.split(delim_split, line[acol])
            data = {
                "title": line[tcol],
                "paperAbstract": line[abscol],
                "authors": authors,
                "startSubmissionId": line[scol],
            }
            submissions.append(data)
            submission_map[line[scol]] = i
            print(json.dumps(data), file=f)

    if args.bid_in:
        raise NotImplementedError("Processing bids not implemented yet")
