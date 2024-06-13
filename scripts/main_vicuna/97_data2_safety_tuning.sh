#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path ./models/defense/3_vicuna-13b-v1.5_safety_training \
    --defense_type safety_tuning \
    --attack AdvPrompter \
    --instructions_path ./data/MaliciousInstruct_100.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_safety_tuning \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b \
    2>&1 | tee -ai ./exp_logs/main_vicuna/MaliciousInstruct_100/safety_tuning/AdvPrompter/main_vicuna_safety_tuning/14_advprompter_vicuna_split0_MaliciousInstruct_100_safety_tuning_AdvPrompter_3_vicuna-13b-v1.5_safety_training_$(date +\%Y\%m\%d_\%H\%M\%S).txt


