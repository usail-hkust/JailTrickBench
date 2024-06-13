#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AmpleGCG \
    --instructions_path ./data/MaliciousInstruct_100.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense \
    --attack_source vicuna \
    2>&1 | tee -ai ./exp_logs/main_vicuna/MaliciousInstruct_100/None_defense/AmpleGCG/main_vicuna_none_defense/8_ample_vicuna_split0_MaliciousInstruct_100_None_defense_AmpleGCG_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt


