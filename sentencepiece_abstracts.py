import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=scratch/abstracts.txt --model_prefix=scratch/abstracts.sp.20k --vocab_size=20000 --character_coverage=0.9995 --hard_vocab_limit=false')
sp = spm.SentencePieceProcessor()
sp.Load('scratch/abstracts.sp.20k.model')

f = open('scratch/abstracts.txt', 'r')
lines = f.readlines()
output = []
for i in lines:
    i = i.strip().lower()
    s0 = sp.EncodeAsPieces(i)
    s0 = " ".join(s0)
    output.append(s0)

fout = 'scratch/abstracts.20k.sp.txt'
fout = open(fout, "w")
for i in output:
    fout.write(i + "\n")
fout.close()


"""
spm.SentencePieceTrainer.Train('--input=scratch/abstracts.txt --model_prefix=scratch/abstracts.sp.10k --vocab_size=10000 --character_coverage=0.9995 --hard_vocab_limit=false')
sp = spm.SentencePieceProcessor()
sp.Load('scratch/abstracts.sp.10k.model')

f = open('scratch/abstracts.txt', 'r')
lines = f.readlines()
output = []
for i in lines:
    i = i.strip().lower()
    s0 = sp.EncodeAsPieces(i)
    s0 = " ".join(s0)
    output.append(s0)

fout = 'scratch/abstracts.10k.sp.txt'
fout = open(fout, "w")
for i in output:
    fout.write(i + "\n")
fout.close()



spm.SentencePieceTrainer.Train('--input=scratch/abstracts.txt --model_prefix=scratch/abstracts.sp.5k --vocab_size=5000 --character_coverage=0.9995 --hard_vocab_limit=false')
sp = spm.SentencePieceProcessor()
sp.Load('scratch/abstracts.sp.5k.model')

f = open('scratch/abstracts.txt', 'r')
lines = f.readlines()
output = []
for i in lines:
    i = i.strip().lower()
    s0 = sp.EncodeAsPieces(i)
    s0 = " ".join(s0)
    output.append(s0)

fout = 'scratch/abstracts.5k.sp.txt'
fout = open(fout, "w")
for i in output:
    fout.write(i + "\n")
fout.close()
"""
