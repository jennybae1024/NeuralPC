#!/bin/bash
#SBATCH --job-name  topic_nli-train-electra
#SBATCH --time      10:00:00
#SBATCH -c          2
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

conda activate tods
ml cuda

dataset_name=("utt2topicWord")
datafile_name="topicNLI"
data_dir="/home/jennybae/data"
sess_dir="/home/jennybae/npc_outputs/"
plm_type="facebook/bart-large-mnli"
num_exp=${#dataset_name[@]}

for ((i==0; i<${num_exp}; i++));
do
    python ../train_nli.py \
        --dataset_name ${dataset_name[i]} \
        --datafile_name ${datafile_name} \
	--data_dir ${data_dir} \
        --model_name_or_path ${plm_type} \
        --output_dir ${sess_dir}/${datafile_name}/${dataset_name[i]}/${plm_type} \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --dev_at_step 3000 \
        --lr 1e-5 \
        --num_epochs 5 \
        --seed 10 \
        --fp16 1
done
