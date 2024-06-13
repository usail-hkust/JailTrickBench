#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path ./models/defense/1_vicuna-13b-v1.5_adv_traning \
    --defense_type adv_training_noaug \
    --attack GCG \
    --instructions_path ./data/MaliciousInstruct_100.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_adv_training_noaug \
    2>&1 | tee -ai ./exp_logs/main_vicuna/MaliciousInstruct_100/adv_training_noaug/GCG/main_vicuna_adv_training_noaug/12_split0_MaliciousInstruct_100_adv_training_noaug_GCG_1_vicuna-13b-v1.5_adv_traning_$(date +\%Y\%m\%d_\%H\%M\%S).txt


