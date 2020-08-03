#PYTHON_BIN=/home/work/miniconda3/bin/python
#source /home/work/miniconda3/bin/activate pytorch-0.3
#PYTHON_BIN='/home/work/miniconda3/envs/pytorch-0.3/bin/python'
source /home/work/anaconda3/bin/activate pytorch-1.1
PYTHON_BIN=/home/work/anaconda3/envs/pytorch-1.1/bin/python
#date=0318
#date=0705
date=0711
#track="Semantics"


rm "./output/softconf_upload_with_detail_by_manager/start-assignments_pc_${date}.txt"
#for track in "Semantics__Short"
for track in `cat ./u_tracks.txt`
do
    ${PYTHON_BIN} softconf_format.py \
       --suggestion_file "output/suggestion_${track}_${date}.jsonl" \
       --track ${track} \
       --date ${date}
done
