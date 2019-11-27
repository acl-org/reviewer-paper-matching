import json
import argparse
from sacremoses import MosesTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--infile", help="input json file")
parser.add_argument("--outfile", help="output file of text, 1 per line")

args = parser.parse_args()

with open(args.infile, "r") as f:
    data = [json.loads(x) for x in f]

entok = MosesTokenizer(lang='en')

abstracts = []
for i in data:
    abstracts.append(i['paperAbstract'])

outfile = open(args.outfile, 'w')
for i in abstracts:
    i = i.strip()
    text = entok.tokenize(i, escape=False)
    text = " ".join(text).lower()
    outfile.write(text + "\n")
