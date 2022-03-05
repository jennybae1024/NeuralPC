#!/bin/bash
#SBATCH --job-name  train_neuralpc-GPT2-NoTopics
#SBATCH --time      4:00:00
#SBATCH -c          2
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

conda activate tods
ml cuda

model="gpt2"
datafile="Prsn2Dialog_DialogLevel"
data_dir="/home/jennybae/data/neuralpc"
sess_dir="/home/jennybae/outputs/neuralpc"
num_exp=${#datafile[@]}


for ((i==0; i<${num_exp}; i++));
do
        python ../train_neuralpc_gpt2.py \
          --data_dir ${data_dir} \
          --datafile_name ${datafile} \
          --model_name_or_path ${model} \
          --output_path ${sess_dir}/${datafile}/${model} \
          --num_epochs 20 \
          --batch_size 8 \
          --dev_at_step 400 \
          --gradient_accumulation_steps 2 \
          --temperature 0.95 \
          --top_k 5 \
          --fp16
done

