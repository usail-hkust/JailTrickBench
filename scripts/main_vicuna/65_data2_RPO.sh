#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type RPO \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/MaliciousInstruct_100.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_rpo \
    2>&1 | tee -ai ./exp_logs/main_vicuna/MaliciousInstruct_100/RPO/PAIR/main_vicuna_rpo/10_split0_MaliciousInstruct_100_RPO_PAIR_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt


