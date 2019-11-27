import io
import sentencepiece as spm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from model_utils import Example, unk_string
from sacremoses import MosesTokenizer

def get_sequences(p1, p2, model, params, fr0=0, fr1=0):
    wp1 = Example(p1)
    wp2 = Example(p2)

    if fr0==1 and fr1==1 and not model.share_vocab:
        wp1.populate_embeddings(model.vocab_fr, model.zero_unk, params.ngrams)
        wp2.populate_embeddings(model.vocab_fr, model.zero_unk, params.ngrams)
        if len(wp1.embeddings) == 0:
            wp1.embeddings.append(model.vocab_fr[unk_string])
        if len(wp2.embeddings) == 0:
            wp2.embeddings.append(model.vocab_fr[unk_string])
    elif fr0==0 and fr1==1 and not model.share_vocab:
        wp1.populate_embeddings(model.vocab, model.zero_unk, params.ngrams)
        wp2.populate_embeddings(model.vocab_fr, model.zero_unk, params.ngrams)
        if len(wp1.embeddings) == 0:
            wp1.embeddings.append(model.vocab[unk_string])
        if len(wp2.embeddings) == 0:
            wp2.embeddings.append(model.vocab_fr[unk_string])
    else:
        wp1.populate_embeddings(model.vocab, model.zero_unk, params.ngrams)
        wp2.populate_embeddings(model.vocab, model.zero_unk, params.ngrams)
        if len(wp1.embeddings) == 0:
            wp1.embeddings.append(model.vocab[unk_string])
        if len(wp2.embeddings) == 0:
            wp2.embeddings.append(model.vocab[unk_string])

    return wp1, wp2

def get_correlation(model, f, params, tok1, tok2, fr0=0, fr1=0):
    f = io.open(f, 'r', encoding='utf-8')
    lines = f.readlines()
    preds = []
    golds = []
    seq1w = []
    seq2w = []

    ct = 0
    for n,i in enumerate(lines):

        i = i.split("\t")
        if len(i) != 3 or len(i[-1].strip()) == 0:
            continue

        p1, p2, score = (i[0], i[1], float(i[2]))
        p1 = " ".join(tok1.tokenize(p1, escape=False))
        p2 = " ".join(tok2.tokenize(p2, escape=False))
        p1 = p1.lower()
        p2 = p2.lower()
        if model.sp is not None:
            p1 = model.sp.EncodeAsPieces(p1)
            p2 = model.sp.EncodeAsPieces(p2)
            p1 = " ".join(p1)
            p2 = " ".join(p2)

        wX1, wX2 = get_sequences(p1, p2, model, params, fr0, fr1)

        seq1w.append(wX1)
        seq2w.append(wX2)

        ct += 1
        if ct % 100 == 0:
            wx1, wl1 = model.torchify_batch(seq1w)
            wx2, wl2 = model.torchify_batch(seq2w)
            scores = model.scoring_function(wx1, wl1, wx2, wl2, fr0, fr1)
            preds.extend(scores.data.cpu().numpy().tolist())
            seq1w = []
            seq2w = []

        golds.append(score)

    if len(seq1w) > 0:
        wx1, wl1 = model.torchify_batch(seq1w)
        wx2, wl2 = model.torchify_batch(seq2w)
        scores = model.scoring_function(wx1, wl1, wx2, wl2, fr0, fr1)
        preds.extend(scores.data.cpu().numpy().tolist())

    return pearsonr(preds, golds)[0], spearmanr(preds, golds)[0]

def evaluate(model, params):
    assert not model.training

    entok = MosesTokenizer(lang='en')

    f = "STS/STS17-test/STS.input.track5.en-en.txt"
    p,s = get_correlation(model, f, params, entok, entok, fr0=0, fr1=0)
    return "track5.en-en.txt\tpearson: {:.3f}\tspearman: {:.3f}".format(p*100, s*100)
