import sys
import json

# Read in a JSON file, grab the abstract, then add every 2 words to each side
for line in sys.stdin:
    words = json.loads(line)['paperAbstract'].split()
    left, right = [], []
    for i, word in enumerate(words):
        (left if i % 4 < 2 else right).append(word)
    print(' '.join(left)+'\t'+' '.join(right))