#!/bin/bash
#SBATCH --job-name  dnli_mnli
#SBATCH --time      10:00:00
#SBATCH -c          2
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

conda activate tods
ml cuda

dataset_name=("utt2topicWord_mnli" "utt2topicWord" "utt2topicWord_ExceptJob")
datafile_name="topicNLI"
data_dir="/home/jennybae/data"
sess_dir="/home/jennybae/npc_outputs/"
plm_type="facebook/bart-large-mnli"
num_exp=${#dataset_name[@]}


for ((i==0; i<${num_exp}; i++));
do
    python ../src/evaluate_nli.py \
        --dataset_name ${dataset_name[i]} \
        --datafile_name ${datafile_name} \
	--data_dir ${data_dir} \
	--model_name_or_path ${plm_type} \
        --checkpoint  ${sess_dir}/${datafile_name}/${dataset_name[i]}/${plm_type} \
        --output_dir ${sess_dir}/${datafile_name}/${dataset_name[i]}/${plm_type} \
        --seed 10 \
        --fp16 1
done

