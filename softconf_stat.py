import sys
import pandas
import json


submissions = []
for lines in sys.stdin:
    submissions.append(json.loads(lines.strip()))

full_submission = pandas.read_csv("./scratch/Submission_Information.csv", skipinitialspace=True, quotechar='"', encoding = "UTF-8")

sid2status = {}
for sid, status in zip(full_submission["Submission ID"], full_submission["Acceptance Status"]):
    if 'Reject' in status:
        continue
    elif 'long' in status.lower():
        sid2status[str(sid)] = 'Long'
    elif 'short' in status.lower():
        sid2status[str(sid)] = 'Short'
    else:
        print('status error!')
        sys.exit(1)

score_stat = {}
review_stat = {}
for submission in submissions:
    sid = str(submission['startSubmissionId'])
    track = submission["Track"]
    if track not in score_stat:
        score_stat[track] = {"similarPapers":[], "assignedReviewers":[], "topSimReviewers": [], "top15SimReviewers": []}
        review_stat[track] = {}
    #submission["similarPapers"].sort(key = lambda x: x["score"], reverse=True)
    for key in score_stat[track]:
        submission[key].sort(key = lambda x:x["score"], reverse=True)
        score_stat[track][key].append([x["score"] for x in submission[key]])
    for review in submission["assignedReviewers"]:
        startid = review["startUsername"]
        if startid not in review_stat[track]:
            review_stat[track][startid] = 0.0
        review_stat[track][startid] += 1.0

def get_stat(input_list):
    input_list.sort()
    avg = sum(input_list)/float(len(input_list))
    Q1 = input_list[int(len(input_list) * 0.25)]
    Q2 = input_list[int(len(input_list) * 0.50)]
    Q3 = input_list[int(len(input_list) * 0.75)]
    return [avg, input_list[0], Q1, Q2, Q3, input_list[-1]]

tracks = list(score_stat.keys())
tracks.sort()
stat_str = 'Avg\tMin\tQ1\tQ2\tQ3\tMax'

print('Track\tassigned Reviewer Num\tavg Paper Per Reviewer\t%s' % ('\t'.join([stat_str for x in range(6)])))
for track in tracks:
    reviewer_num = len(review_stat[track])
    avg_paper_num = sum([v for (k, v) in review_stat[track].items()])/ len(review_stat[track])
    res = [avg_paper_num]
    for i in range(3):
        res += get_stat([x[i] for x in score_stat[track]["assignedReviewers"]])
    for key in ["similarPapers", "topSimReviewers", "top15SimReviewers"]:
        res += get_stat([sum(x)/len(x) for x in score_stat[track][key]])
    print('%s\t%d\t%s' % (track, reviewer_num, '\t'.join([str(round(x, 2)) for x in res])))

