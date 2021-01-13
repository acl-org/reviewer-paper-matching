import sys
from collections import defaultdict

import numpy as np


def calc_reviewer_id_mapping(reviewers, author_col):
    assert(author_col == 'name' or author_col == 'id')
    reviewer_id_map = defaultdict(lambda: [])
    author_col_plural = author_col+'s'
    for j, x in enumerate(reviewers):
        for reviewer in x[author_col_plural]:
            reviewer_id_map[reviewer].append(j)
    return reviewer_id_map


def calc_reviewer_db_mapping(reviewers, db, author_field='authors', warn_under=1):
    """ Calculate correspondence between reviewers and papers

    :param reviewers: A list of reviewer names, or reviewer IDs
    :param db: A DB with papers, and a field `author_field` for authors
    :param author_field: The field to look for in the DB
    :param warn_under: Throw a warning if a reviewer has fewer papers than this value
    :return: a numpy array with rows as reviewers, columns as entries in the DB
    """
    print(f'Calculating reviewer-paper mapping for {len(reviewers)} reviewers and {len(db)} papers', file=sys.stderr)
    reviewer_id_map_name = calc_reviewer_id_mapping(reviewers, 'name')
    reviewer_id_map_id = calc_reviewer_id_mapping(reviewers, 'id')

    mapping = np.zeros( (len(db), len(reviewers)) )
    for i, entry in enumerate(db):
        for cols in entry[author_field]:
            js = []
            author_col = 'id'
            reviewer_id_map = reviewer_id_map_id
            if author_col in cols:
                if cols[author_col] in reviewer_id_map:
                    js = reviewer_id_map[cols[author_col]]
            else:
                for x in cols[author_col + 's']:
                    if x in reviewer_id_map:
                        js.extend(reviewer_id_map[x])

            author_col = 'name'
            reviewer_id_map = reviewer_id_map_name
            if author_col in cols: 
                if cols[author_col] in reviewer_id_map:
                    js = reviewer_id_map[cols[author_col]]
            else:
                for x in cols[author_col + 's']:
                    if x in reviewer_id_map:
                        js.extend(reviewer_id_map[x])

            for j in js:
                mapping[i, j] = 1
    num_papers = mapping.sum(axis=0)
    for rev, num in zip(reviewers, num_papers):
        name = rev['names'][0]
        # this print spams too many warnings and does not allow to notice really important ones
        # if num < warn_under:
        #     print(f'WARNING: Reviewer {name} ({rev["startUsername"]}) has {num} papers in the database', file=sys.stderr)
    return mapping


def print_text_report(query, file):
    print('----------------------------------------------', file=file)
    print('*** Paper Title', file=file)
    print(query['title'], file=file)
    print('*** Paper Abstract', file=file)
    print(query['paperAbstract'], file=file)
    print('\n*** Similar Papers', file=file)

    for x in query['similarPapers']:
        my_title, my_abs, my_score = x['title'], x['paperAbstract'], x['score']
        print(f'# {my_title} (Score {my_score})\n{my_abs}', file=file)
    print('', file=file)
    print('\n*** Best Matched Reviewers', file=file)
    for x in query['topSimReviewers']:
        my_name, my_score = x['names'][0], x['score']
        print(f'# {my_name} (Score {my_score})', file=file)
    print('\n*** Assigned Reviewers', file=file)
    for x in query['assignedReviewers']:
        my_name, my_score = x['names'][0], x['score']
        print(f'# {my_name} (Score {my_score})', file=file)
    print('', file=file)


def print_progress(i, mod_size):
    if i != 0 and i % mod_size == 0:
        sys.stderr.write('.')
        if int(i/mod_size) % 50 == 0:
            print(i, file=sys.stderr)
        sys.stderr.flush()
