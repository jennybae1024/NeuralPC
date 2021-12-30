#!/bin/bash
#SBATCH --job-name  dnli_mnli
#SBATCH --time      10:00:00
#SBATCH -c          2
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user m6r7a8t8o8y7f1b5@milabsnu.slack.com

#conda activate tods
#ml cuda

dataset_name="utt2topicWord"
train_datafile="RawPrsn_TurnLevel_Topic20_GE"
generator="facebook-bart-large"
generated_data_type=("TestType2_TurnLevel_mscPersona" "TestType3_TurnLevel_Yahoo" "TestType3_TurnLevel_YahooReddit")

datafile_name="topicNLI"
data_dir="/home/jennybae/data"
sess_dir="/home/jennybae/outputs/"
plm_type="facebook/bart-large-mnli"
num_exp=${#generated_data_type[@]}


for ((i==0; i<${num_exp}; i++));
do
    python ../evaluate_nli.py \
        --dataset_name ${dataset_name} \
	--train_datafile ${train_datafile} \
	--generator ${generator} \
	--generated_data_type ${generated_data_type[i]} \
	--data_dir ${data_dir} \
	--model_name_or_path ${plm_type} \
        --checkpoint  ${sess_dir}/${datafile_name}/${dataset_name}/${plm_type} \
        --output_dir ${sess_dir}/${datafile_name}/${dataset_name}/${plm_type} \
        --seed 10 \
        --fp16 1
done


