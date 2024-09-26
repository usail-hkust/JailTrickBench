#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # gpt-3.5-turbo or gpt-4

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."

    if [ ! -d "../results/cache" ]; then
        mkdir "../results/cache"
        echo "Folder '../results/cache' created."
    fi
fi
# data_offset for starting from a certain point 
for data_offset in 0
do

    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="../results/attack_on_${model}" \
        --config.n_train_data=520 \
        --config.data_offset=$data_offset \
        --config.noun_sub=True \
        --config.verb_sub=True \
        --config.noun_wordgame=True \
        --config.suffix=True \
        --config.topk_sub=5 \
        --config.general_reconstruction=True \
        --config.prompt_info_path="../../attack_prompt_data/gpt_automated_processing_results/prompts_information.json" 

done