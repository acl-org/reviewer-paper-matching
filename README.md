# ACL Reviewer Matching Code

This is an initial pass at code to match reviewers for the [ACL Conferences](http://aclweb.org).
It is based on paper matching between abstracts of submitted papers and a database of papers from
[Semantic Scholar](https://www.semanticscholar.org).

## Usage Instructions

### Installing and Training Model (before review process)

**Step 1:** install requirements:

    pip install -r requirements.txt
    
**Step 2:** Download the semantic scholar dump
[following the instructions](http://api.semanticscholar.org/corpus/download/).

**Step 3:** Wherever you downloaded the semantic scholar dump, make a virtual link to the `s2` directory here.

    ln -s /path/to/semantic/scholar/dump s2

**Step 4:** Make a scratch directory where we'll put working files

    mkdir -p scratch/
    
**Step 5:** For efficiency purposes, we'll filter down the semantic scholar dump to only papers where it has a link to
aclweb.org, a rough approximation of the papers in the ACL Anthology.

    zcat s2/s2-corpus*.gz | grep aclweb.org > scratch/acl-anthology.json

**Step 6:** Train a model of semantic similarity based on the ACL anthology (this step requires GPU). Alternatively,
you may contact the authors to get a model distributed to you.

    bash download_sts17.sh
    python tokenize_abstracts.py --infile scratch/acl-anthology.json --outfile scratch/abstracts.txt
    python sentencepiece_abstracts.py --infile scratch/abstracts.txt --vocab-size 20000 \
                                      --model-name scratch/abstracts.sp.20k --outfile scratch/abstracts.20k.sp.txt 
    python -u train_similarity.py --data-file scratch/abstracts.20k.sp.txt \
                                  --model avg --dim 1024 --epochs 20 --ngrams 0 --share-vocab 1 --dropout 0.3 \
                                  --outfile scratch/similarity-model.pt --batchsize 64 --megabatch-size 1 \
                                  --megabatch-anneal 10 --seg-length 1 \
                                  --sp-model scratch/abstracts.sp.20k.model 2>&1 | tee scratch/training.log

### Creating Assignments (during review process)

**Step 1:** In START, go to the "Spreadsheet Maker" and download CSV spreadsheets for "Submissions"
and "Author/Reviewer Profiles" saving them to `scratch/Submission_Information.csv` and
`scratch/Profile_Information.csv`.

Natalie has hard-coded (line 50) a file path to u_tracks.txt, a file with track names.  It is included in the repo.  (Exact same one as for COI detection.)  Please do change this to something smarter.

**Step 2:** Process these files into the jsonl or npy format used by this software:

    python softconf_extract.py \
        --profile_in=scratch/Profile_Information.csv \
        --submission_in=scratch/Submission_Information.csv \
        --reviewer_out=scratch/reviewers.jsonl \
        --submission_out=scratch/submissions.jsonl \
        --bid_out=scratch/cois.npy \
		--tracks_file=tracks.txt \
		--committee_list=committee_list_[track]_[date].p \
		--current_track=[track]

**Step 3:** Create and save the reviewer assignments:

    python suggest_reviewers.py \
        --submission_file=scratch/submissions.jsonl \
        --db_file=scratch/acl-anthology.json \
        --reviewer_file=scratch/reviewers.jsonl \
        --model_file=scratch/similarity-model.pt \
        --max_papers_per_reviewer=5 \
        --reviews_per_paper=3 \
        | tee scratch/assignments.jsonl

You will then have assignments written to both the terminal and `scratch/assignments.jsonl`. You can modify
`reviews_per_paper` and `max_papers_per_reviewer` to change the number of reviews assigned to each paper and max number
of reviews per reviewer. After you've output the suggestions, you can also print them (and save them to a file
`scratch/assignments.txt`) in an easier-to-read format by running:

    python suggest_to_text.py < scratch/assignments.jsonl | tee scratch/assignments.txt
    
**Step 4:** If you want to turn these into START format to re-enter them into START, you can run the following
command:

    python softconf_package.py < scratch/assignments.jsonl > scratch/start-assignments.csv
    
then import `scratch/start-assignments.csv` into start using the data import interface.

### Evaluating Assignments (to test the model)

**Step 1:** In order to evaluate the system, you will need either (a) a conference with gold-standard bids, or (b)
some other way to create information about bids automatically.

**Step 1a:** If you have gold-standard bids, in START, go to the "Spreadsheet Maker" and download CSV spreadsheets for
"Submissions," "Author/Reviewer Profiles," and "Bids" saving them to `scratch/Submission_Information.csv` and
`scratch/Profile_Information.csv`, and `scratch/Bid_Information.csv`. Then run the following:

    python softconf_extract.py \
        --profile_in=scratch/Profile_Information.csv \
        --submission_in=scratch/Submission_Information.csv \
        --bids_in=scratch/Bid_Information.csv \
        --reviewer_out=scratch/reviewers.jsonl \
        --submission_out=scratch/submissions.jsonl \
        --bid_out=scratch/bids.npy
        
**Step 1b:** If you do not have this information from START, do the same as above but without `--bids_in`. Then you
will have to create a numpy array `bids.npy` where rows correspond to submissions, and columns correspond to the bids
of potential reviewers (0=COI, 1=no, 2=maybe, 3=yes).

**Step 2:** Follow the steps in the previous section on reviewer assignment to generate `assignments.jsonl`.

**Step 3:** Evaluate the assignments using the following command:

        python evaluate_suggestions.py \
            --suggestion_file=scratch/assignments.jsonl \
            --reviewer_file=scratch/reviewers.jsonl \
            --bid_file=scratch/bids.npy

This will tell you the ratio of assignments that were made to each bid, where in general a higher ratio of "3"s is
better.

## Method Description

The method for suggesting reviewers works in three steps. First, it uses a model to calculate the perceived similarity
between paper abstracts for the submitted papers and papers that the reviewers have previously written. Then, it
aggregates the similarity scores for each paper into an overall similarity score between the reviewer and the paper.
Finally, it performs a reviewer assignment, assigning reviewers to each paper given constraints on the maximimum number
of papers per reviewer.

### Similarity Between Paper Abstracts

The method for measuring similarity of paper abstracts is based on the subword or character averaging method described
in the following paper:

    @inproceedings{wieting19simple,
        title={Simple and Effective Paraphrastic Similarity from Parallel Translations},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url={https://arxiv.org/abs/1909.13872},
        year={2019}
    }

We trained a model on abstracts from the ACL anthology to attempt to make the similarity model as topically appropriate
as possible. After running this model, this gives us a large matrix of similarity scores between submitted papers, and
papers where at least one of the reviewers in the reviewer list is an author on the paper.

### Reviewer Score Aggregation

Next, we aggregate the paper-paper matching scores into paper-reviewer matching scores. Assume that `S_r` is the set of
all papers written by a reviewer `r`, and `p` is a submitted paper. The paper-reviewer matching score of this paper
can be calculated in a number of ways, but currently this code uses the following by default:

    match(p,r) = max(match(S_r,p)) + second(match(S_r,p)) / 2 + third(match(S_r,p)) / 3

Where `max`, `second`, and `third` indicate the reviewer's best, second-best, and third-best matching paper's scores,
respectively. The motivation behind this metric is based on the intuition that we would both like a reviewer to have
written at least one similar paper (hence the `max`) and also have experience in the field moving beyond a single paper,
as the one paper they have written might be an outlier (hence the `second` and `third`).

The code for this function, along with several other alternatives, can be found in `suggest_reviewers.py`'s 
`calc_aggregate_reviewer_score` function.

### Reviewer Assignment

Finally, the code suggests an assignment of reviewers. This is done by trying to maximize the overall similarity score
for all paper-reviewer matches, under the constraint that each paper should have exactly `reviews_per_paper` reviewers
and each reviewer should review a maximum of `max_papers_per_reviewer` reviews. This is done by solving a linear program
that attempts to maximize the sum of the reviewer-paper similarity scores based on these constraints.

# Credits

This code was written/designed by Graham Neubig, John Wieting, Arya McCarthy, and Amanda Stent.

