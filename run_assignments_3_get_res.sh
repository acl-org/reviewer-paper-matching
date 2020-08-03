source /home/work/anaconda3/bin/activate pytorch-1.1
PYTHON_BIN=/home/work/anaconda3/envs/pytorch-1.1/bin/python
date=0711
#track="Semantics"

from=$1
to=$2
gpu_id=$3
export CUDA_VISIBLE_DEVICES=$gpu_id

#for track in `cat ./u_tracks.txt`
for track in `sed -n "$from, $to p" ./u_tracks.txt`
do
    ${PYTHON_BIN} suggest_reviewers.py \
        --submission_file="./output/submissions_${track}_${date}.jsonl"\
        --db_file="./scratch/acl-anthology.json" \
        --reviewer_file="./output/reviewers_${track}_${date}.jsonl" \
        --model_file="./scratch/similarity-model.pt" \
        --bid_file="./output/cois_${track}_${date}.npy" \
        --max_papers_per_reviewer=3 \
        --reviews_per_paper=3 \
        --suggestion_file="./output/suggestion_${track}_${date}.jsonl" \
        --output_dir="./output/suggest_reviewer/" \
        --track="${track}" \
        --date="${date}" \
        --save_paper_matrix="./output/suggest_reviewer/paper_matrix_${track}_${date}.db" \
        --save_aggregate_matrix="./output/suggest_reviewer/aggregate_matrix_${track}_${date}.db" \
        #--load_paper_matrix="./output/suggest_reviewer/paper_matrix_${track}_${date}.db.npy" \
        #--load_aggregate_matrix="./output/suggest_reviewer/aggregate_matrix_${track}_${date}.db.npy" \
        #--metareviewers="./scratch/computeCOIs/metareviewers_${track}_${date}.npy" \
        #--save_paper_matrix=""
        #| tee output/assignments_${track}_${date}.jsonl
done
