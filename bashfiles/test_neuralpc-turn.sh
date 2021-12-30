#!/bin/bash
#SBATCH --job-name  neuralpc-test-turn
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
train_datafile="RawPrsn_TurnLevel_Topic20_GE"
test_datafile=("TestType2_TurnLevel_mscPersona" "TestType3_YahooReddit" "TestType3_TurnLevel_YahooReddit")
test_type=("2" "3" "3")
max_turns=("12" "12" "12")
turn_window_size=("7" "7")
data_dir="/home/jennybae/data"
sess_dir="/home/jennybae/npc_outputs"
num_exp=${#test_datafile[@]}

for ((i==0; i<${num_exp}; i++));
do
	    python ../generate_neuralpc_turnlvl.py \
	            --datafile_name ${test_datafile[i]} \
	            --test_type ${test_type[i]} \
	            --max_turns ${max_turns[i]} \
	            --turn_window_size ${turn_window_size[i]} \
	            --output_file_name ${test_datafile[i]} \
	            --data_dir ${data_dir} \
	            --output_dir ${sess_dir}/${dataset}/${train_datafile}/${model}  \
	            --batch_size 4 \
	            --max_target_length 512 \
	            --num_beams 1 \
	            --top_k 0 \
	            --top_p 0.98
done
