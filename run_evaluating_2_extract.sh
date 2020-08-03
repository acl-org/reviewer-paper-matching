python suggest_reviewers.py \
    --submission_file=scratch/submissions.jsonl \
    --db_file=scratch/acl-anthology.json \
    --reviewer_file=scratch/reviewers.jsonl \
    --model_file=scratch/similarity-model.pt \
    --max_papers_per_reviewer=5 \
    --reviews_per_paper=3 \
    | tee scratch/assignments.jsonl
