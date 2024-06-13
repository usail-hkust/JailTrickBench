#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type self_reminder \
    --attack GCG \
    --instructions_path ./data/MaliciousInstruct_100.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_self_reminder \
    2>&1 | tee -ai ./exp_logs/main_vicuna/MaliciousInstruct_100/self_reminder/GCG/main_vicuna_self_reminder/9_split0_MaliciousInstruct_100_self_reminder_GCG_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt


