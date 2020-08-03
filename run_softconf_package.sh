#PYTHON_BIN=/home/work/miniconda3/bin/python
#source /home/work/miniconda3/bin/activate pytorch-0.3
#PYTHON_BIN='/home/work/miniconda3/envs/pytorch-0.3/bin/python'
source /home/work/anaconda3/bin/activate pytorch-1.1
PYTHON_BIN=/home/work/anaconda3/envs/pytorch-1.1/bin/python
#date=0318
#date=0705
date=0710
#track="Semantics"


#for track in "Semantics__Short"
for track in `cat ./u_tracks.txt`
do
    ${PYTHON_BIN} softconf_package.py \
       --suggestion_file "output/suggestion_${track}_${date}.jsonl" \
       > output/softconf_upload/start-assignments_${track}_${date}.csv
done
