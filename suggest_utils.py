import sys
from collections import defaultdict

import numpy as np


def calc_reviewer_db_mapping(reviewers, db, author_col='name', author_field='authors', warn_under=1):
    """ Calculate correspondence between reviewers and papers

    :param reviewers: A list of reviewer names, or reviewer IDs
    :param db: A DB with papers, and a field `author_field` for authors
    :param author_col: The column in the author field to check in the DB
    :param author_field: The field to look for in the DB
    :param warn_under: Throw a warning if a reviewer has few papers under this value
    :return: an NP array with rows as reviewers, columns as entries in the DB
    """
    print(f'Calculating reviewer-paper mapping for {len(reviewers)} reviewers and {len(db)} papers', file=sys.stderr)
    mapping = np.zeros( (len(reviewers), len(db)) )
    reviewer_id_map = defaultdict(lambda: [])
    for i, reviewer in enumerate(reviewers):
        reviewer_id_map[reviewer].append(i)
    for j, entry in enumerate(db):
        for cols in entry[author_field]:
            if cols[author_col] in reviewer_id_map:
                for i in reviewer_id_map[cols[author_col]]:
                    mapping[i,j] = 1
    num_papers = mapping.sum(axis=1)
    for name, num in zip(reviewers, num_papers):
        if num < warn_under:
            print(f'WARNING: Reviewer {name} has {num} papers in the database', file=sys.stderr)
    return mapping