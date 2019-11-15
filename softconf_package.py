import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suggestion_file", type=str, required=True, help="The reviewer CSV file"
    )

    args = parser.parse_args()

    # Load the data
    with open(args.suggestion_file, "r") as f:
        submissions = [json.loads(x) for x in f]

    for submission in submissions:
        sid = submission["startId"]
        print(f"{sid}:" + ";".join([x["startId"] for x in submission["authors"]]))
