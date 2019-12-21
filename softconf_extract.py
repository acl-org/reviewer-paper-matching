# NS has changed this to be track dependent: now for all reviewers not in the input track, we put 0s (i.e., COI) for the entire column.
# I filter to include only the submissions (i.e., in the submissions and bids files) for the specific track)
# I also adjust to allow AC assignment only

import argparse
import json
import csv
import suggest_utils
import sys
import numpy as np
import re
import pandas
import os.path
import pickle

def find_colids(colnames, header):
    colids, colmap = [-1 for _ in colnames], {}
    for i, k in enumerate(colnames):
        for alias in k.split('|'):
            colmap[alias] = i
    for i, col in enumerate(header):
        if col in colmap:
            colids[colmap[col]] = i
    if any([x == -1 for x in colids]):
        print(f'WARNING: Couldn\'t find column ids in {colnames} {colids}', file=sys.stderr)
    return colids

def validate_track_name(tn):
    with open('tracks.txt','r') as infile:
        tracks=set([l.strip() for l in infile.readlines()])
        if tn in tracks:
            return True
    return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_in", type=str, default=None, help="The submission CSV file")
    parser.add_argument("--profile_in", type=str, required=True, help="The profile CSV file")
    parser.add_argument("--bid_in", type=str, default=None, help="The submission numpy file")
    parser.add_argument("--reviewer_out", type=str, required=True, help="The track-specific reviewer json file")
    parser.add_argument("--submission_out", type=str, default=None, help="The track-specific submission json file")
    parser.add_argument("--bid_out", type=str, default=None, help="The track-specific submission numpy file")
    parser.add_argument("--tracks_file", type=str, default='tracks.txt', help="The list of tracks")
    parser.add_argument("--committee_list", type=str, help="ordered list of committee members--same order as bids columns")
    parser.add_argument("--track", type=str, help='current track')

    args = parser.parse_args()
    
    if os.path.exists(args.tracks_file):
        with open(args.tracks_file,'r') as tfile:
            with open('/home/natalie/projects/acl/reviewer-coi-detection-acl2020/data/u_tracks.txt','r') as t2file:
                tracks=[x.strip() for x in tfile.readlines()]
                t2list=[x.strip() for x in t2file.readlines()]
                tdict=dict(zip(t2list,tracks))
    else:
        print('No tracks file.  Exiting...')

    # Get framework for bids
#    bids = np.full( (len(csv_lines)-1, len(reviewers)), 2 ) # Initialize with "Maybes"
    if args.bid_in:
        bids=np.load(args.bid_in)
    else:
        print('Bids file missing.')
        sys.exit()
        
    # Process reviewers
    #Track managers (SAC) have, "Is a track manager" = "Yes"
    #ACs have, "Is a track manager" = "No", and "Roles"="committee:[trackname]"
    #Reviewers have "Is a track manager" = "No", and "Roles"="[trackname]"
    #Here we remove people/papers from tracks in the assignment process by putting a 0 in their bids column
    with open(args.profile_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Username', 'Email', 'First Name', 'Last Name', 'Semantic Scholar ID', 'Roles', 'level', 'Is a track manager'] #NS added level
    ucol, ecol, fcol, lcol, scol, rcol, level_col, tman = find_colids(colnames, csv_lines[0])
    reviewers, reviewer_map, profile_map = [], {}, {}
    usernames=[]
    emails=[]
    with open(args.reviewer_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]): #iterating through profiles
            s2id = line[scol].split('/')[-1]
            # Author data
            data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col]), 'Roles':line[rcol], 'Is a track manager':line[tman]}
            profile_map[line[ucol]] = data
            profile_map[line[ecol]] = data
            
#            #if 'committee' in line[rcol]:
            if tdict[args.track] in line[rcol]:
                if 'manager' not in line[rcol]:
                    if True:
                    #if (args.ac_assignment and 'committee' in line[rcol]) or ((not args.ac_assignment) and 'committee' not in line[rcol]): #ac OR reviewer
                        rev_data = {'name': f'{line[fcol]} {line[lcol]}', 'ids': [s2id], 'startUsername': line[ucol], 'level':float(line[level_col]), 'Roles':line[rcol], 'Is a track manager':line[tman]}
                        print(json.dumps(rev_data), file=f)
                        reviewer_map[line[ucol]] = len(reviewers)
                        reviewer_map[line[ecol]] = len(reviewers)
                        reviewers.append(rev_data)

    # Process submissions (if present)
    if not args.submission_in:
        print('No submission file. Exiting...')
        sys.exit(0)
    with open(args.submission_in, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        csv_lines = list(csv_reader)
    colnames = ['Submission ID', 'Title', 'Abstract|Summary', 'Authors', 'All Author Emails', 'Track']
    scol, tcol, abscol, acol, ecol, trcol = find_colids(colnames, csv_lines[0])
    submissions, submission_map = [], {}

    #load committee
    committee_list=pickle.load(open(args.committee_list,'rb'))
    #print('committee_list', committee_list)

    # Write submissions
    missing=0
    delim_re = re.compile(r'(?:, | and )')
    with open(args.submission_out, 'w') as f:
        for i, line in enumerate(csv_lines[1:]): #order of submissions
            author_emails = line[ecol].split('; ')
            author_names = re.split(delim_re, line[acol])
            authors = []
            for ae, an in zip(author_emails, author_names):
                if ae in profile_map:
                    authors.append(profile_map[ae])
#                    if ae in reviewer_map:
#                        bids[i,reviewer_map[ae]] = 0
#                    missing+=1
                else:
                    #print(f'WARNING: could not find account for {an}, just using name {an}')
                    authors.append({'name': an, 'ids': []})
                    missing+=1
                    
            data = {'title': line[tcol], 'paperAbstract': line[abscol], 'authors': authors, 'startSubmissionId': line[scol], 'Track':line[trcol]}
            submissions.append(data)
            submission_map[line[scol]] = i
            print(json.dumps(data), file=f)
            
    #Track managers (SAC) have, "Is a track manager" = "Yes"
    #ACs have, "Is a track manager" = "No", and "Roles"="committee:[trackname]"
    #Reviewers have "Is a track manager" = "No", and "Roles"="[trackname]"
            num_reviewers=0
            num_acs=0
            num_sacs_managers=0
            #print(committee_list)
            for j in range(len(committee_list)):
                reviewer=committee_list[j]
#                print(reviewer, end=' ')
                #if reviewer not in reviewer_map:
                #print('reviewer', reviewer)
                #print(reviewers[reviewer_map[reviewer]])
                #print(reviewers[reviewer_map[reviewer]]['Roles'])
                if reviewer in reviewer_map and 'manager' in reviewers[reviewer_map[reviewer]]['Roles']: #SAC, PC chair, GC
                    bids[i,j] = 0
                    num_sacs_managers+=1
                    #print('manager', reviewer, num_sacs_managers)
                #elif reviewer in reviewer_map and args.ac_assignment: #AC assignments only
                #    if 'committee' not in reviewers[reviewer_map[reviewer]]['Roles']: 
                #        bids[i,j] = 0
                #        print(reviewers[reviewer_map[reviewer]]['Roles'])
                #        num_acs+=1
                #        print('ac',reviewer,num_acs)
                #    else:
                #        num_reviewers+=1
                #        print('reviewer',reviewer,num_reviewers)
                #elif reviewer in reviewer_map:    #reviewer assignments only
                #    if 'committee' in reviewers[reviewer_map[reviewer]]['Roles']:
                #        bids[i,j] = 0
                #        num_acs+=1
                #        print('ac',reviewer,num_acs)
                #    else:
                #        num_reviewers+=1
                #        print('reviewer',reviewer,num_reviewers)
                #        print(reviewers[reviewer_map[reviewer]]['Roles'] )
                #        print(line[trcol])
                        #for track in tracks:
                        #    if not (track in reviewer_map[reviewer]['Roles'] and track in line[trcol]):
                        #        bids[i,j] = 0 #i=submission row, j=reviewer
                        #        print(0,end='')
                        #    else:
                        #        print(2,end='')
                        #print()
#            sys.exit()

    print('Authors missing',missing)
    if args.bid_out:
        with open(args.bid_out, 'wb') as f:
            np.save(f, bids)
    
    #print(bids)
