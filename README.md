# ACL Reviewer Matching Code

This is an initial pass at code to match reviewers for the [ACL Conferences](http://aclweb.org).
It is based on paper matching between abstracts of submitted papers and a database of papers from
[Semantic Scholar](https://www.semanticscholar.org).

## Installing

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

**Step 6:** In the scratch directory, create two files of paper abstracts `scratch/abstracts.txt` and reviewer names
`scratch/reviewers.txt`.

**Step 7:** Train a model of semantic similarity based on the ACL anthology.

TODO: finish up directions

## Method Description

This method works by calculating the similarity between papers, then finding reviewers that have several (by default
up to 3) papers similar to the submitted one. Lower-ranked papers are slightly downweighted, allow for reviewers with
even one highly relevant paper to get more highly matched.

The method for measuring similarity of paper abstracts is based on the token embedding averaging method in the following
paper:

    @inproceedings{wieting19simple,
        title={Simple and Effective Paraphrastic Similarity from Parallel Translations},
        author={Wieting, John and Gimpel, Kevin and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Association for Computational Linguistics},
        url={https://arxiv.org/abs/1909.13872},
        year={2019}
    }

# Credits

This code was written/designed by Graham Neubig, John Wieting, Arya McCarthy, and Amanda Stent.

