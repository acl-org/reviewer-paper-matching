import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()

parser.add_argument("--infile", help="name of input file (tokenized, 1 text per line)")
parser.add_argument("--outfile", help="name of output file processed by sentencepiece")
parser.add_argument("--model-name", help="sentencepiece model name")
parser.add_argument("--vocab-size", help="sentencepiece vocabulary size")

args = parser.parse_args()

spm.SentencePieceTrainer.Train('--input={0} --model_prefix={1} --vocab_size={2} --character_coverage=0.9995 '
                               '--hard_vocab_limit=false'.format(args.infile, args.model_name, args.vocab_size))
sp = spm.SentencePieceProcessor()
sp.Load(args.model_name)

f = open(args.infile, 'r')
lines = f.readlines()
output = []
for i in lines:
    i = i.strip().lower()
    s0 = sp.EncodeAsPieces(i)
    s0 = " ".join(s0)
    output.append(s0)

fout = args.outfile
fout = open(fout, "w")
for i in output:
    fout.write(i + "\n")
fout.close()
