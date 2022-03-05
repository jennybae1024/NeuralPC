#!/bin/bash
#SBATCH --job-name  neuralpc-train-bart_large
#SBATCH --time      20:00:00
#SBATCH -c          4
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

conda activate tods
ml cuda

model="facebook/bart-large"
dataset="neuralpc"
datafile="Prsn2Dialog_DialogLevel"
data_dir="/home/jennybae/data"
sess_dir="/home/jennybae/npc_outputs"

python ../train_neuralpc_bart.py \
  --datafile_name ${datafile} \
  --data_dir ${data_dir} \
  --model_name_or_path ${model} \
  --output_path ${sess_dir}/${dataset}/${datafile}/${model} \
  --per_gpu_train_batch_size 8 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 1000 \
  --fp16
