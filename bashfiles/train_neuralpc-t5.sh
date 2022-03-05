#!/bin/bash
#SBATCH --job-name  train_neuralpc-T5-dialogLevel-NoTopics
#SBATCH --time      4:00:00
#SBATCH -c          2
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

conda activate tods
ml cuda

model="t5-base"
dataset="neuralpc"
datafile="Prsn2Dialog_DialogLevel"
data_dir="/home/jennybae/data/neuralpc"
sess_dir="/home/jennybae/outputs/neuralpc"

python ../train_neuralpc_t5.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_eval \
    --dataset_name $dataset \
    --train_file ${data_dir}/${datafile}_train.csv \
    --validation_file ${data_dir}/${datafile}_valid.csv \
    --output_dir ${sess_dir}/${datafile}/${model} \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --max_source_length 256 \
    --val_max_target_length 512 \
    --max_target_length 512 \
    --predict_with_generate \
    --pad_to_max_length \
    --save_steps 1500 \
    --fp16 True \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --logging_steps 100
