#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type self_reminder \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_self_reminder \
    2>&1 | tee -ai ./exp_logs/main_vicuna/harmful_bench_50/self_reminder/PAIR/main_vicuna_self_reminder/2_split0_harmful_bench_50_self_reminder_PAIR_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt


