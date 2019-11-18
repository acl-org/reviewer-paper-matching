#!/bin/bash
set -e

for epoch in 11 16 6 1; do
  for minrev in 0 3; do
    for maxrev in 8 10; do
      for agg in max weighted_top2 weighted_top3 weighted_top5; do

        id=e$epoch-agg$agg-min$minrev-max$maxrev
        
        if [[ ! -d output/$id ]]; then
        
          mkdir output/$id
          
          echo "python -u suggest_reviewers.py --submission_file=data/emnlp2019-curated.json --db_file=scratch/acl-anthology.json --reviewer_file=data/acl2020-area-chair-nameids.json --model_file=scratch/similarity-model.pt_$epoch.pt --max_papers_per_reviewer=$maxrev --min_papers_per_reviewer=$minrev --output_type=json --suggestion_file=output/$id/suggest.json --bid_file=data/aclrev-emnlppap.npy --aggregator=$agg &> output/$id/suggest.log"
          python -u suggest_reviewers.py --submission_file=data/emnlp2019-curated.json --db_file=scratch/acl-anthology.json --reviewer_file=data/acl2020-area-chair-nameids.json --model_file=scratch/similarity-model.pt_$epoch.pt --max_papers_per_reviewer=$maxrev --min_papers_per_reviewer=$minrev --output_type=json --suggestion_file=output/$id/suggest.json --bid_file=data/aclrev-emnlppap.npy --aggregator=$agg &> output/$id/suggest.log
          
          echo "python -u evaluate_suggestions.py --suggestion_file output/$id/suggest.json --reviewer_file=data/acl2020-area-chair-nameids.json --bid_file data/aclrev-emnlppap.npy &> output/$id/eval.log"
          python -u evaluate_suggestions.py --suggestion_file output/$id/suggest.json --reviewer_file=data/acl2020-area-chair-nameids.json --bid_file data/aclrev-emnlppap.npy &> output/$id/eval.log
        
        else
          echo "output/$id already exists"
        fi

      done
    done
  done
done

