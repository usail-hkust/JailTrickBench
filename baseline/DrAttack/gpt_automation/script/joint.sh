#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="../gpt_generate_content.py"

# Define arguments
PROMPT_PATH="./data/advbench/harmful_behaviors.csv"
MODEL="gpt-4"
GENERATE_MODE="joint"
SAVE_PATH="./attack_prompt_data/gpt_automated_processing_results/joint_info.json"

# Execute the Python script with the arguments
python $PYTHON_SCRIPT --prompt_path $PROMPT_PATH --model $MODEL --generate_mode $GENERATE_MODE --save_path $SAVE_PATH