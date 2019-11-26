import json
from sacremoses import MosesTokenizer

with open('scratch/acl-anthology.json', "r") as f:
    data = [json.loads(x) for x in f]

entok = MosesTokenizer(lang='en')

abstracts = []
for i in data:
    abstracts.append(i['paperAbstract'])

#print(len(abstracts))
for i in abstracts:
    i = i.strip()
    text = entok.tokenize(i, escape=False)
    text = " ".join(text).lower()
    #abstracts.append(text)
    print(text)
    #print(len(abstracts))
#print(len(abstracts))
