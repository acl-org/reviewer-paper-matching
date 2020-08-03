PYTHON_BIN=/home/work/anaconda3/envs/pytorch-1.1/bin/python

date=0711
#track="Semantics"
for track in `cat ./u_tracks.txt`;
do
    ${PYTHON_BIN} softconf_extract.py \
        --profile_in="./scratch/Profile_collectCOIs.csv" \
        --submission_in="./scratch/Submission_Information.csv" \
        --bid_in="./scratch/computeCOIs/coibids_${track}_${date}.npy" \
        --reviewer_out="output/reviewers_${track}_${date}.jsonl" \
        --submission_out="output/submissions_${track}_${date}.jsonl" \
        --submission_to_bid="./scratch/computeCOIs/submission_to_bid.txt" \
        --bid_out="./output/cois_${track}_${date}.npy" \
        --tracks_file=tracks.txt \
        --committee_list="./scratch/computeCOIs/committee_list_${track}_${date}.p" \
        --track="${track}"
done
